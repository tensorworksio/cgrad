# CGRAD
![logo](docs/logo.png)

Intended to be a Torch-like autograd engine, inspired by [micrograd](https://github.com/karpathy/micrograd/tree/master)

## Dependencies
```bash
add-apt-repository ppa:snaipewastaken/ppa
apt-get update
apt-get install libcriterion-dev
apt-get install libcsptr-dev
apt-get install meson ninja-build
```

## How to use it
```C
#include "tensor.h"
#include "ops.h"
#include "log.h"

int main()
{
    log_set_level(LOG_INFO);

    tensor_t *a = tensor((float[]){2., 4., 6.}, (int[]){3}, 1, true);
    tensor_t *b = tensor((float[]){1., 2., 0.}, (int[]){3}, 1, true);
    // c = a + b
    tensor_t *c = tensor_add(a, b);
    // c = c - 1
    c = tensor_sub_tf(c, 1.);
    // d = c ** 3
    tensor_t *d = tensor_pow_tf(c, 3.);
    // e = relu(d)
    tensor_t *e = tensor_relu(d);
    // f = sum(e)
    tensor_t *f = tensor_sum(e);

    tensor_backward(f); // compute gradients

    tensor_print(a, PRINT_ALL); // print tensors a.data and a.grad = d(f)/d(a)
    tensor_print(b, PRINT_ALL); // print tensors b.data and b.grad = d(f)/d(b)

    // recursively free all tensors in the graph
    tensor_free(f, true);
    return 0;
}
```

## Compile and run

### With meson & ninja
```bash
mkdir build && cd build
meson setup ..
ninja && meson test
./main
```

### With Make
```bash
make && make test
./main
```

## TODO

### Bug
- tensor with > 1 parent segfault when using recursive tensor_free -> use smalloc for tensor as well

### Features
- polymorphic iterator range_iterator >< array_iterator 
- Reduce operator for specific axes
- Matmul operator as a combination of sum, add and reshape ideally
- Pooling operators
- Conv operator as a specific case of pool operator
