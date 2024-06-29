# TeaNet binary model

## Prepare external libraries

Please see `README.md` in `deploy/external`

## How to build binary model

To build a model for production,

```
make
```

For development,

```
make dev
```

By default, `make` uses `deploy/install` directory as install directory of external libraries. If you have external libraries in different directory, you can specify that like this:

```
make DEPLOY_INSTALL_PREFIX=/opt/pfn/pfp/deploy
```
