# Silero

Rust implementation of [Silero VAD](https://github.com/snakers4/silero-vad).

## Running Tests

Tests require all the features to be present as serde deserializations are used
for snapshot testing. Also, test data is stored via git lfs so you'll need to use
git lfs to clone the repository to run the tests.

To run tests do:

```
cargo test --all-features
```

## License

This code is licensed under the terms of the MIT License. See LICENSE for more
details.
