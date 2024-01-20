# CGRAD

Intended to be a Torch-like autograd engine, inspired by [micrograd](https://github.com/karpathy/micrograd/tree/master)

## Dependencies
Install criterion to run tests:
```bash
apt-get install libcriterion-dev
```
## How to use it
```C
#include "tensor.h"
#include "ops.h"
#include "log.h"

int main() {
    log_set_level(LOG_INFO);

    tensor_t* a = tensor((float[]){2., 4., 6.}, (int[]){3}, 1, true);
    tensor_t* b = tensor((float[]){1., 2., 0.}, (int[]){3}, 1, true);
    // c = a + b
    tensor_t* c = tensor_add(a, b);
    // c = c - 1
    c = tensor_sub_tf(c, 1.);
    // d = c ** 3
    tensor_t* d = tensor_pow_tf(c, 3.);
    // e = relu(d)
    tensor_t* e = tensor_relu(d);
    // f = sum(e)
    tensor_t* f = tensor_sum(e);

    tensor_backward(f); // compute gradients
    
    tensor_print(a); // print tensors a.data and a.grad = d(f)/d(a)
    tensor_print(b); // print tensors b.data and b.grad = d(f)/d(b)

    // recursively free all tensors in the graph
    tensor_free(f, true);
    return 0;
}
```

## Compile and run

```bash
make && ./main
```

## Run test

```bash
make test
```

## TODO

### Bug
- print_data misses the \n logic: stupid problem I swear
- Anonymous tensor Segfault when using recursive tensor_free

### Features
- Slice operator for tensor: 
    - needs more testing 
    - needs some logic to handle negative indexing
- Reduce operator for specific axes (require slices)
- Matmul operator as a combination of sum, add and reshape ideally
- Pooling operators
- Conv operator as a specific case of pool operator
