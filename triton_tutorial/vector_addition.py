import torch
import triton
import triton.language as tl
DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)                         # asks "which program am I?" (
    block_start = pid * BLOCK_SIZE                      # where this program's slice begins. For pid=2, that's 2 * 4 = 8.
    offsets = block_start + tl.arange(0, BLOCK_SIZE)    # the actual memory indices this program touches. tl.arange(0,4) is [0,1,2,3], so offsets = [8,9,10,11] (top row of the second diagram). Note this is a vector of indices, not a single number — that's the whole point of a block.
    mask = offsets < n_elements                         # the boolean guard, [True, True, False, False], that disables the out-of-bounds lanes.
    x = tl.load(x_ptr + offsets, mask=mask)             # x_ptr is the address of the tensor's first element, and x_ptr + offsets gives one address per lane. The mask tells load to fetch only where it's safe.
    y = tl.load(y_ptr + offsets, mask=mask)             
    output = x + y                                      # add elementwise
    tl.store(output_ptr + offsets, output, mask=mask)   # write results back at the same offsets, again gated by mask so slots 10 and 11 are never written.

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)                                    # Allocates the output buffer upfront. empty_like is used instead of zeros_like because zeroing would be wasted work                                                    
    assert x.device == DEVICE and \
        y.device == DEVICE and \
        output.device == DEVICE                                     # As kernel does raw pointer arithmetic into GPU memory, if any tensor were sitting on the CPU, those addresses would be meaningless and you'd get corruption or a crash.
    n_elements = output.numel()                                     # This single number gets passed into the kernel so each program knows where the real data ends.
    grid = lambda meta: (
        triton.cdiv(n_elements, meta['BLOCK_SIZE']), 
    )                                                               # Decides how many programs to lauch. cdiv is ceiling division: cdiv(98432, 1024) = 97. Written as a lambda rather than just 97 because the grid depends on the BLOCK_SIZE, and Triton passes the acutal block size in through the meta dictionary at launch time. The tariling comma, a one-element tuple, signals a 1D grid
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)     # [grid] tells Triton to spawn 97 program instances, each getting a distinct pid. 
    return output                                                   # The kernel launch is asynchronous -- the CPU may reach this return before the GPU has finished.

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x,y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True)
