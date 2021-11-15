# A cached version of native_functions.yml

Each version of PyTorch will use a new file, when we move to another version of pytorch run 

```
wget -O native_functions.{VERSION}.yml https://raw.githubusercontent.com/pytorch/pytorch/release/{VERSION}/aten/src/ATen/native/native_functions.yaml
```

with {VERSION} replaced with the version of pytorch (i.e 1.10)