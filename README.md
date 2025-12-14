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
#include "forward.h"
#include "log.h"
#include "tensor.h"

int
main ()
{
    log_set_level (LOG_INFO);

    smart tensor_t *a = tensor ((float[]) { 1., 2., 3., 4., 5., 6. }, (int[]) { 2, 3 }, 2, true);
    smart tensor_t *b = tensor ((float[]) { 1., 2., 3., 4., 5., 6. }, (int[]) { 3, 2 }, 2, true);
    smart tensor_t *c = tensor_matmul (a, b);
    smart tensor_t *f = tensor_sum (c, NULL, 0);
    tensor_backward (f); // compute gradients

    tensor_print (a, PRINT_ALL); // print tensors a.data and a.grad = d(f)/d(a)
    tensor_print (b, PRINT_ALL); // print tensors b.data and b.grad = d(f)/d(b)
    tensor_print (c, PRINT_ALL); // print tensors c.data and c.grad = d(f)/d(c)
    tensor_print (f, PRINT_ALL); // print tensors f.data and f.grad = d(f)/d(f)

    return 0;
}
```

## Run tests
```bash
meson test -C build --wrap='valgrind --leak-check=full --error-exitcode=1' --verbose
```

## TODO
- Mul operator to auto broadcast shape like in numpy
- Reduce operators: max, min, mean, prod
- Pooling operators
- Conv operator as a specific case of pool operator
