# pytriton-load-test-example

Build container.
```sh
docker build -t pytriton-load-test-example-transformer .
```

Start container in interactive mode.
```sh
docker run -it -p8000:8000 -p8001:8001 -p8002:8002 -v ./non-existent-path:/models -v ./pytriton_examples:/examples pytriton-load-test-example-transformer bash
```

```sh
docker run --gpus=1 -it -p8000:8000 -p8001:8001 -p8002:8002 -v ./non-existent-path:/models -v ./pytriton_examples:/examples pytriton-load-test-example-transformer bash
```


Run command within the container.
```sh
python3 ./transformer/multiple_model_single_record_server.py
```
