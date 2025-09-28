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
#include "log.h"
#include "forward.h"
#include "tensor.h"

int
main ()
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 2., 4., 6. }, (int[]) { 3 }, 1, true);
    smart tensor_t *b = tensor ((float[]) { 1., 2., 0. }, (int[]) { 3 }, 1, true);
    // c = a + b
    smart tensor_t *c = tensor_add (a, b);
    // c = c - 1
    TENSOR_REBIND (c, tensor_sub_tf (c, 1.0f));
    // d = c ** 3
    smart tensor_t *d = tensor_pow_tf (c, 3.);
    // e = relu(d)
    smart tensor_t *e = tensor_relu (d);
    // f = sum(e)
    smart tensor_t *f = tensor_sum (e);

    tensor_backward (f); // compute gradients

    tensor_print (a, PRINT_ALL); // print tensors a.data and a.grad = d(f)/d(a)
    tensor_print (b, PRINT_ALL); // print tensors b.data and b.grad = d(f)/d(b)

    return 0;
}
```

## Run tests
```bash
meson test -C build --wrap='valgrind --leak-check=full --error-exitcode=1' --verbose
```

## TODO
- Reduce operator for specific axes (require slices)
- Matmul operator as a combination of sum, add and reshape ideally
- Pooling operators
- Conv operator as a specific case of pool operator
