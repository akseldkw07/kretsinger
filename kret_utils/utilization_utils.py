import psutil
import torch


class UtilizationUtils:
    @classmethod
    def print_gpu_utilization(cls):
        """Print GPU utilization (CUDA devices only)"""
        if not torch.cuda.is_available():
            print("CUDA not available")
            return

        print("\n=== GPU Utilization (CUDA) ===")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = props.total_memory / 1024**3

            print(f"GPU {i}: {props.name}")
            print(f"  Memory Allocated: {allocated:.2f} GB / {total:.2f} GB ({allocated/total*100:.1f}%)")
            print(f"  Memory Reserved:  {reserved:.2f} GB / {total:.2f} GB ({reserved/total*100:.1f}%)")
            print(f"  Memory Free:      {total - reserved:.2f} GB")

    @classmethod
    def print_mps_utilization(cls):
        """Print MPS (Apple Silicon) utilization"""
        if not torch.backends.mps.is_available():
            print("MPS not available")
            return

        print("\n=== MPS Utilization (Apple Silicon) ===")

        # MPS memory stats
        if hasattr(torch.mps, "current_allocated_memory"):
            allocated = torch.mps.current_allocated_memory() / 1024**3
            print(f"  Current Allocated Memory: {allocated:.2f} GB")

        if hasattr(torch.mps, "driver_allocated_memory"):
            driver_allocated = torch.mps.driver_allocated_memory() / 1024**3
            print(f"  Driver Allocated Memory:  {driver_allocated:.2f} GB")

        # System memory (approximation for unified memory)
        vm = psutil.virtual_memory()
        total_ram = vm.total / 1024**3
        used_ram = vm.used / 1024**3
        available_ram = vm.available / 1024**3

        print(f"\n  System Memory (Unified):")
        print(f"    Total:     {total_ram:.2f} GB")
        print(f"    Used:      {used_ram:.2f} GB ({vm.percent:.1f}%)")
        print(f"    Available: {available_ram:.2f} GB")

    @classmethod
    def print_cpu_utilization(cls):
        """Print CPU utilization"""
        print("\n=== CPU Utilization ===")

        # CPU count
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        print(f"  Physical Cores: {physical_cores}")
        print(f"  Logical Cores:  {logical_cores}")

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1, percpu=False)
        print(f"  Overall Usage:  {cpu_percent:.1f}%")

        # Per-core usage
        per_cpu = psutil.cpu_percent(interval=1, percpu=True)
        print(f"  Per-Core Usage:")
        for i, usage in enumerate(per_cpu):
            bar = "█" * int(usage / 5) + "░" * (20 - int(usage / 5))
            print(f"    Core {i:2d}: {bar} {usage:5.1f}%")

        # CPU frequency
        freq = psutil.cpu_freq()
        if freq:
            print(f"\n  Frequency:")
            print(f"    Current: {freq.current:.0f} MHz")
            if freq.min > 0:
                print(f"    Min:     {freq.min:.0f} MHz")
            if freq.max > 0:
                print(f"    Max:     {freq.max:.0f} MHz")
