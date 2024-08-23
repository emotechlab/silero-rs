# Silero

Rust implementation of [Silero VAD](https://github.com/snakers4/silero-vad).

## Running Tests

Tests require all the features to be present as serde deserializations are used
for snapshot testing. Also, test data is stored via git lfs so you'll need to use
git lfs to clone the repository to run the tests.

To run tests do:

```text
cargo test --all-features
```

If a test fails we have a script in `scripts/plot_audio.py` which will plot out
a number of insights from the generated files. You can run this on the existing
snapshots and ones generated from your test and try to see why the behaviour
differs. In some instances the difference may be desired, but in others it will
be symptomatic of a bug or issue in the code. Please justify in the PR text
whenever you update a snapshot file!

## License

This code is licensed under the terms of the MIT License. See LICENSE for more
details.
