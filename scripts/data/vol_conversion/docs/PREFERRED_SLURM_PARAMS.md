# Preferred Slurm Parameters for This Cluster

## Standard Configuration

```bash
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=<job_name>_%j.out
#SBATCH --error=<job_name>_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohaina@bc.edu
```

## Key Parameters

- **Partition**: `weilab` (cluster-specific partition)
- **CPUs**: `4` (standard allocation)
- **Memory**: `64G` (generous allocation for data processing)
- **Time**: `20:00:00` (20 hours - adjust based on job needs)
- **Module**: `miniconda` (for Python environments)
- **Email**: `gohaina@bc.edu` (user-specific)

## Usage Examples

### Quick Test Jobs
```bash
#SBATCH --time=00:20:00  # 20 minutes
#SBATCH --mem=32G        # Less memory for quick tests
```

### Long-Running Jobs
```bash
#SBATCH --time=20:00:00  # 20 hours
#SBATCH --mem=64G        # Full memory allocation
```

### High-Memory Jobs
```bash
#SBATCH --mem=128G       # Double memory if needed
#SBATCH --cpus-per-task=8  # More CPUs for parallel processing
```

## Notes

- Always use `weilab` partition for this cluster
- `miniconda` module is preferred over `miniforge`
- 64GB memory is standard for data processing tasks
- 4 CPUs provide good balance of resources
- 20-hour time limit allows for long-running conversions
