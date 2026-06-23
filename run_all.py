#!/usr/bin/env python3
"""
CARLA Data Collection Supervisor with Auto-Restart.
Safely manages only its own CARLA and collection processes by PID.
Supports multiple parallel instances on different ports.
"""
import argparse
import gc, os, queue as _queue, re, sys
import signal
import subprocess
import threading
import time
from datetime import datetime

import psutil

# Import shared utility from collect_data
from collect_data import save_load_route_fail

# Try to import carla for connection verification
try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


class CARLASupervisor:
    """Supervisor for CARLA data collection with automatic crash recovery."""

    def __init__(self, port=3100, max_restarts=50, lidar_channels=32, townid="05", \
                 start_id=0,
                 carla_dir="/home/kin/data/CARLA/CARLA_0.9.16",
                 collect_dir="/home/kin/data/CARLA/CARLA_0.9.16/PythonAPI/opensf-carla",
                 data_dir="/home/kin/data/CARLA/data-32-160k-2k"):
        self.port = port
        self.lidar_channels = lidar_channels
        self.townid = townid
        self.carla_dir = carla_dir
        self.collect_dir = collect_dir
        self.data_dir = data_dir
        self.start_id = start_id

        # CARLA server command
        self.carla_cmd = [
            "./CarlaUE4.sh",
            "--quality-level=Epic",
            f"-carla-rpc-port={self.port}",
            "-RenderOffScreen"
        ]
        
        # Collection script command
        self.collect_cmd = self._build_collect_cmd()
        
        # Process handles
        self.carla_proc = None
        self.collect_proc = None
        self.carla_pid = None
        self.collect_pid = None
        
        # State tracking
        self.restart_count = 0
        self.max_restarts = max_restarts
        self.last_route_id = -1
        self.checkpoint_file = f"outputs/restart_checkpoint_port{self.port}.txt"
        
        # Memory management: restart all processes every N successful routes
        self.routes_since_restart = 0
        self.max_routes_before_restart = 25  # Restart every 25 routes to prevent memory leak
        
        # Monitoring state (use lock for thread safety)
        self._lock = threading.Lock()
        self._monitoring = False
        self._carla_crashed = False
        self._ram_exceeded = False
        
        # RAM monitoring settings (in GB)
        self.max_ram_gb = 55  # Kill all processes if RAM exceeds this
        self.ram_check_interval = 60*5  # Check RAM every N seconds

        # Stall detection: if collect_data.py produces no stdout for this long, assume hung
        self.stall_timeout_secs = 20 * 60  # 20 minutes
        
        # Log file
        self.log_file = f"outputs/supervisor_port{self.port}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        os.makedirs("outputs", exist_ok=True)
    
    def _build_collect_cmd(self):
        """Build collection command based on lidar configuration."""
        points_per_second = 160000 if self.lidar_channels == 32 else 460000
        return [
            "python", "collect_data.py",
            f"sensors.lidar_semantic.channels={self.lidar_channels}",
            f"sensors.lidar_semantic.points_per_second={points_per_second}",
            "simulation.route_id=-1",
            f"simulation.port={self.port}",
            # "world_settings.logging_level=DEBUG",
            f"simulation.route_file={self.collect_dir}/assets/data/town{self.townid}.xml",
            f"simulation.data_output={self.data_dir}",
            f"simulation.start_id={self.start_id}",
        ]
    
    def log(self, msg):
        """Write log message to both console and file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[Port {self.port}][{timestamp}] {msg}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def load_checkpoint(self):
        """Load last processed route from checkpoint."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    self.last_route_id = int(f.read().strip())
                self.log(f"Loaded checkpoint: last route was {self.last_route_id}")
            except (ValueError, IOError):
                self.log("Failed to load checkpoint, starting from beginning")
    
    def save_checkpoint(self, route_id):
        """Save current route to checkpoint."""
        with open(self.checkpoint_file, 'w') as f:
            f.write(str(route_id))
    
    def _kill_process_tree(self, pid):
        """Kill a process and all its children by PID."""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            
            # Terminate children first
            for child in children:
                try:
                    self.log(f"Terminating child process {child.pid}")
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # Wait for children to terminate
            _, alive = psutil.wait_procs(children, timeout=3)
            
            # Force kill remaining children
            for p in alive:
                try:
                    self.log(f"Force killing child process {p.pid}")
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
            
            # Terminate parent
            try:
                self.log(f"Terminating parent process {pid}")
                parent.terminate()
                parent.wait(timeout=3)
            except psutil.TimeoutExpired:
                self.log(f"Force killing parent process {pid}")
                parent.kill()
            except psutil.NoSuchProcess:
                pass
                
        except psutil.NoSuchProcess:
            self.log(f"Process {pid} already terminated")
        except Exception as e:
            self.log(f"Error killing process tree {pid}: {e}")
    
    def _kill_carla(self):
        """Kill only OUR CARLA process by PID."""
        if self.carla_pid:
            self.log(f"Killing CARLA process {self.carla_pid} on port {self.port}...")
            self._kill_process_tree(self.carla_pid)
            self.carla_pid = None
            self.carla_proc = None
            time.sleep(2)
            # Force garbage collection to release any carla client references
            gc.collect()
    
    def _kill_existing_carla_on_port(self):
        """Kill any existing CARLA process using our port (from previous sessions)."""
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.laddr.port == self.port and conn.status == 'LISTEN':
                    try:
                        proc = psutil.Process(conn.pid)
                        proc_name = proc.name().lower()
                        # Only kill if it looks like CARLA
                        if 'carla' in proc_name or 'ue4' in proc_name:
                            self.log(f"Found stale CARLA process {conn.pid} on port {self.port}, killing...")
                            self._kill_process_tree(conn.pid)
                            time.sleep(2)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        except (psutil.AccessDenied, OSError) as e:
            self.log(f"Warning: Could not check for existing processes on port: {e}")
    
    def _kill_collect(self):
        """Kill only OUR collection script process by PID."""
        if self.collect_pid:
            self.log(f"Killing collection process {self.collect_pid}...")
            self._kill_process_tree(self.collect_pid)
            self.collect_pid = None
            self.collect_proc = None
            time.sleep(2)
            # Force garbage collection to release memory held by subprocess references
            gc.collect()
    
    def _start_carla(self):
        """Start CARLA server."""
        self.log("Starting CARLA server...")
        
        # Kill any stale CARLA process on our port from previous sessions
        self._kill_existing_carla_on_port()
        
        try:
            # IMPORTANT: Don't capture stdout to avoid pipe buffer blocking!
            # CARLA outputs a lot of logs, which can fill the pipe buffer
            # and cause CARLA to hang waiting for the buffer to be read.
            self.carla_proc = subprocess.Popen(
                self.carla_cmd,
                cwd=self.carla_dir,
                stdout=subprocess.DEVNULL,  # Discard stdout to prevent blocking
                stderr=subprocess.DEVNULL,  # Discard stderr too
            )
            self.carla_pid = self.carla_proc.pid
            self.log(f"CARLA started with PID: {self.carla_pid}")
            
            # Wait a minimum time before checking - fresh CARLA takes at least 10-15s to start
            # This prevents connecting to a stale instance that might still be shutting down
            time.sleep(10)
            
            # Wait for CARLA to be ready by actually trying to connect
            if not self._wait_for_carla_ready(max_wait=60, check_interval=2):
                self.log("ERROR: CARLA failed to become ready!")
                self._kill_carla()
                return False
            
            self.log("CARLA server is ready and accepting connections")
            return True
        except Exception as e:
            self.log(f"Failed to start CARLA: {e}")
            self.carla_pid = None
            return False
    
    def _wait_for_carla_ready(self, max_wait=60, check_interval=2):
        """
        Wait for CARLA to be ready by attempting to connect.
        
        Args:
            max_wait: Maximum time to wait in seconds
            check_interval: Time between connection attempts
        
        Returns:
            True if CARLA is ready, False if timeout or process died
        """
        if not CARLA_AVAILABLE:
            # Fallback to simple wait if carla module not available
            self.log("carla module not available, using fixed wait time...")
            time.sleep(max_wait // 2)
            return self.carla_proc.poll() is None
        
        start_time = time.time()
        attempt = 0
        
        while time.time() - start_time < max_wait:
            # Check if CARLA process is still alive
            if self.carla_proc.poll() is not None:
                self.log(f"CARLA process died during startup (exit code: {self.carla_proc.returncode})")
                return False
            
            attempt += 1
            client = None
            world = None
            try:
                # Try to connect to CARLA
                client = carla.Client('localhost', self.port)
                client.set_timeout(5.0)
                
                # Try to get the world to verify server is fully ready
                world = client.get_world()
                if world is not None:
                    self.log(f"CARLA ready after {attempt} attempts ({time.time() - start_time:.1f}s)")
                    return True
                    
            except Exception as e:
                elapsed = time.time() - start_time
                self.log(f"Waiting for CARLA... attempt {attempt} ({elapsed:.0f}s): {type(e).__name__}")
            finally:
                # CRITICAL: Release carla client/world references to prevent memory leak
                # These objects hold references to CARLA's shared memory even after server dies
                del world
                del client
            
            time.sleep(check_interval)
        
        self.log(f"CARLA did not become ready within {max_wait} seconds")
        return False
    
    def _start_collection(self):
        """Start data collection script."""
        self.log("Starting data collection script...")
        try:
            self.collect_proc = subprocess.Popen(
                self.collect_cmd,
                cwd=self.collect_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            self.collect_pid = self.collect_proc.pid
            self.log(f"Collection script started with PID: {self.collect_pid}")
            return True
        except Exception as e:
            self.log(f"Failed to start collection script: {e}")
            self.collect_pid = None
            return False
    
    def _monitor_carla_thread(self):
        """Background thread to monitor CARLA server health."""
        while True:
            with self._lock:
                if not self._monitoring:
                    break
            
            if self.carla_proc and self.carla_proc.poll() is not None:
                returncode = self.carla_proc.returncode
                self.log(f"CARLA server died with exit code {returncode}")
                with self._lock:
                    self._carla_crashed = True
                break
            time.sleep(1)
    
    def _monitor_ram_thread(self):
        """Background thread to monitor RAM usage and trigger emergency restart if exceeded."""
        while True:
            with self._lock:
                if not self._monitoring:
                    break
            
            try:
                # Get total RAM usage in GB
                mem = psutil.virtual_memory()
                used_gb = mem.used / (1024 ** 3)
                
                if used_gb > self.max_ram_gb:
                    self.log(f"EMERGENCY: RAM usage {used_gb:.1f}GB exceeds {self.max_ram_gb}GB threshold!")
                    with self._lock:
                        self._ram_exceeded = True
                    break
            except Exception as e:
                self.log(f"Error monitoring RAM: {e}")
            
            time.sleep(self.ram_check_interval)
    
    def _monitor_collection(self):
        """Monitor both collection script and CARLA server."""
        self.log("Monitoring collection script and CARLA server...")

        # Read stdout in a separate thread so readline() never blocks the main loop.
        # This allows stall detection: if no output arrives within stall_timeout_secs,
        # we can detect a hung process instead of waiting forever.
        stdout_lines = _queue.Queue()

        def _read_stdout():
            try:
                for line in iter(self.collect_proc.stdout.readline, ''):
                    stdout_lines.put(line)
            finally:
                stdout_lines.put(None)  # sentinel: process exited

        with self._lock:
            self._monitoring = True
            self._carla_crashed = False
            self._ram_exceeded = False

        carla_thread = threading.Thread(target=self._monitor_carla_thread, daemon=True)
        carla_thread.start()

        ram_thread = threading.Thread(target=self._monitor_ram_thread, daemon=True)
        ram_thread.start()

        reader_thread = threading.Thread(target=_read_stdout, daemon=True)
        reader_thread.start()

        route_pattern = re.compile(r"Processing Route ID: (\d+)")
        route_done_pattern = re.compile(r"Route \d+ completed successfully")
        completed_pattern = re.compile(r"Data collection completed for all routes")

        try:
            while True:
                # Check crash/RAM flags before blocking on next line
                with self._lock:
                    if self._carla_crashed:
                        self._monitoring = False
                        self.log(f"CARLA server crashed at route {self.last_route_id}")
                        return "CARLA_CRASH"
                    if self._ram_exceeded:
                        self._monitoring = False
                        self.log(f"RAM exceeded threshold at route {self.last_route_id}, triggering emergency restart...")
                        return "RAM_EXCEEDED"

                try:
                    line = stdout_lines.get(timeout=self.stall_timeout_secs)
                except _queue.Empty:
                    with self._lock:
                        self._monitoring = False
                    self.log(
                        f"No output for {self.stall_timeout_secs // 60}min at route "
                        f"{self.last_route_id} — process appears hung. Triggering restart."
                    )
                    return "STALL"

                if line is None:
                    # Sentinel: stdout closed, process has exited
                    with self._lock:
                        self._monitoring = False
                    returncode = self.collect_proc.poll()
                    if returncode == -11:  # SIGSEGV
                        self.log(f"Collection script SEGFAULT at route {self.last_route_id}")
                        return "SEGFAULT"
                    elif returncode == 0:
                        self.log("Collection script exited normally")
                        return "COMPLETED"
                    else:
                        self.log(f"Collection script exited with code {returncode}")
                        return "ERROR"

                print(f"[Port {self.port}] {line}", end='')

                route_match = route_pattern.search(line)
                if route_match:
                    route_id = int(route_match.group(1))
                    self.save_checkpoint(route_id)
                    self.last_route_id = route_id

                if route_done_pattern.search(line):
                    self.routes_since_restart += 1
                    self.log(f"Route completed. Routes since restart: {self.routes_since_restart}/{self.max_routes_before_restart}")
                    if self.routes_since_restart >= self.max_routes_before_restart:
                        with self._lock:
                            self._monitoring = False
                        self.log(f"Reached {self.max_routes_before_restart} routes, triggering memory cleanup restart...")
                        return "MEMORY_RESTART"

                if completed_pattern.search(line):
                    with self._lock:
                        self._monitoring = False
                    self.log("Collection completed successfully!")
                    return "COMPLETED"

        except Exception as e:
            with self._lock:
                self._monitoring = False
            self.log(f"Error monitoring: {e}")
            return "ERROR"
    
    def cleanup(self):
        """Clean up only our own processes."""
        self.log("Cleaning up processes...")
        self._kill_collect()
        self._kill_carla()
        time.sleep(2)
    
    def run(self):
        """Main supervisor loop."""
        self.log("=" * 60)
        self.log(f"CARLA Data Collection Supervisor Started (Port {self.port}) on Channels {self.lidar_channels}")
        self.log(f"Memory management: restart every {self.max_routes_before_restart} routes")
        self.log(f"RAM limit: emergency restart if RAM exceeds {self.max_ram_gb}GB")
        self.log("=" * 60)
        
        self.load_checkpoint()
        
        while self.restart_count < self.max_restarts:
            self.log(f"\n--- Attempt {self.restart_count + 1}/{self.max_restarts} ---")
            
            self.cleanup()
            
            if not self._start_carla():
                self.log("Failed to start CARLA, retrying in 10 seconds...")
                time.sleep(10)
                self.restart_count += 1
                continue
            
            if not self._start_collection():
                self.log("Failed to start collection script")
                self.restart_count += 1
                continue
            
            result = self._monitor_collection()
            
            if result == "COMPLETED":
                self.log("All routes completed successfully!")
                self.cleanup()
                if os.path.exists(self.checkpoint_file):
                    os.remove(self.checkpoint_file)
                return 0
            
            # Handle memory management restart (not a failure, don't increment restart_count)
            if result == "MEMORY_RESTART":
                self.log("Memory cleanup restart - killing collect_data.py and CARLA to free RAM")
                self._kill_collect()  # Kill collect_data.py first!
                self._kill_carla()    # Then kill CARLA
                self.routes_since_restart = 0  # Reset counter
                gc.collect()  # Force garbage collection
                time.sleep(3)
                continue  # Don't increment restart_count
            
            # Handle RAM exceeded - emergency restart with longer wait time
            if result == "RAM_EXCEEDED":
                self.log("EMERGENCY RAM cleanup - killing all processes and waiting 10s...")
                self._kill_collect()
                self._kill_carla()
                self.routes_since_restart = 0
                gc.collect()
                time.sleep(10)  # Longer wait for RAM to be freed
                # Don't increment restart_count for RAM exceeded (it's a recovery, not a failure)
                continue
            
            # Handle failures - record route failure and restart
            if result in ("SEGFAULT", "CARLA_CRASH", "ERROR", "STALL"):
                error_msg = {
                    "SEGFAULT": "Collection script segfault",
                    "CARLA_CRASH": "CARLA server crashed",
                    "STALL": f"Collection hung (no output for {self.stall_timeout_secs // 60}min)",
                    "ERROR": "Unknown error occurred"
                }
                self.log(f"{error_msg[result]}, restarting from route {self.last_route_id + 1}")
                save_load_route_fail(self.townid, self.last_route_id, save=True)
                self.restart_count += 1
                self.routes_since_restart = 0  # Reset counter on failure too
                time.sleep(5)
        
        self.log(f"Max restarts ({self.max_restarts}) reached. Exiting.")
        self.cleanup()
        return 1


# Global reference for signal handler
_supervisor = None


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nReceived interrupt signal, cleaning up...")
    if _supervisor is not None:
        _supervisor.cleanup()
    sys.exit(0)


def main():
    """Main entry point."""
    global _supervisor
    
    parser = argparse.ArgumentParser(description='CARLA Data Collection Supervisor')
    parser.add_argument('-p', '--port', type=int, default=3100, help='CARLA RPC port')
    parser.add_argument('-c', '--channels', type=int, default=32, help='Lidar channels')
    parser.add_argument('-t', '--townids', type=str, default="05",
                        help='Town IDs for route files (comma-separated for multiple, e.g., "05,04,10")')
    parser.add_argument('--carla_dir', type=str, 
                        default="/home/kin/data/CARLA/CARLA_0.9.16",
                        help='Path to CARLA directory')
    parser.add_argument('--collect_dir', type=str,
                        default="/home/kin/data/CARLA/CARLA_0.9.16/PythonAPI/opensf-carla",
                        help='Path to data collection script directory')
    parser.add_argument('-r', '--restart_interval', type=int, default=25,
                        help='Restart all processes every N routes to free memory (default: 25)')
    parser.add_argument('-d', '--data_dir', type=str, default="/home/kin/data/CARLA/data-32-160k-2k",
                        help='Path to data directory')
    parser.add_argument('-m', '--max_restarts', type=int, default=50,
                        help='Maximum number of restarts before giving up (default: 50)')
    parser.add_argument('-s', '--start_id', type=int, default=0,
                        help='Starting route ID (default: 0)')
    parser.add_argument('--max_ram_gb', type=int, default=55,
                        help='Emergency restart if RAM usage exceeds this many GB (default: 55)')
    parser.add_argument('--stall_timeout', type=int, default=20,
                        help='Restart if collect_data.py produces no output for this many minutes (default: 20)')
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    # Parse townids from comma-separated string
    townids = [tid.strip() for tid in args.townids.split(',')]
    
    overall_result = 0
    for i, townid in enumerate(townids):
        print(f"\n{'='*60}")
        print(f"Processing townid {townid} ({i+1}/{len(townids)})")
        print(f"{'='*60}\n")
        
        _supervisor = CARLASupervisor(
            port=args.port,
            lidar_channels=args.channels,
            townid=townid,
            carla_dir=args.carla_dir,
            collect_dir=args.collect_dir,
            data_dir=args.data_dir,
            start_id=args.start_id,
            max_restarts=args.max_restarts
        )
        _supervisor.max_routes_before_restart = args.restart_interval
        _supervisor.max_ram_gb = args.max_ram_gb
        _supervisor.stall_timeout_secs = args.stall_timeout * 60
        result = _supervisor.run()
        
        if result != 0:
            print(f"\nWARNING: townid {townid} failed with exit code {result}")
            overall_result = 1
        else:
            print(f"\nSUCCESS: townid {townid} completed successfully")
        
        # Small delay between townids
        if i < len(townids) - 1:
            time.sleep(5)
    
    return overall_result


if __name__ == "__main__":
    sys.exit(main())