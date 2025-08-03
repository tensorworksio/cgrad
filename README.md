# CGRAD
![logo](docs/logo.png)

Intended to be a Torch-like autograd engine, inspired by [micrograd](https://github.com/karpathy/micrograd/tree/master)

## Dependencies
<details open>
    <summary>Ubuntu</summary>

```bash
add-apt-repository ppa:snaipewastaken/ppa
apt update
apt install python3 ninja-build meson
apt install libcriterion-dev
apt install libcsptr-dev
```

</details>

<details>
    <summary>macOS</summary>

```bash
brew install meson
brew install criterion
brew install libcsptr
export LDFLAGS="-L/opt/homebrew/opt/criterion/lib -L/opt/homebrew/opt/libcsptr/lib"
export CPPFLAGS="-I/opt/homebrew/opt/criterion/include -I/opt/homebrew/opt/libcsptr/include"
```

</details>

## Build
```bash
meson setup --wipe build
meson compile -C build
```

## Usage
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

## Run tests
```bash
meson test -C build # --wrap='valgrind --leak-check=full --error-exitcode=1'
```

## TODO
- Move movement ops to ops & define grad
- tensor_cat requires many children if requires_grad
- Reduce operator for specific axes (require slices)
- Matmul operator as a combination of sum, add and reshape ideally
- Pooling operators
- Conv operator as a specific case of pool operator
