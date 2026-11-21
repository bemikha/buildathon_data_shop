# Shopping Sessions: Rust pipeline.
FROM rust:1-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
# AVX-512/BMI/POPCNT on x86 (the scorer is Cascade Lake or newer); plain build elsewhere.
RUN echo 'if [ "$(uname -m)" = x86_64 ]; then export RUSTFLAGS="-C target-cpu=x86-64-v4"; fi' > /env.sh
# Cache dependencies in their own layer.
RUN . /env.sh && mkdir src && echo "fn main(){}" > src/main.rs && cargo build --release && rm -rf src
COPY src ./src
RUN . /env.sh && touch src/main.rs && cargo build --release

FROM debian:bookworm-slim
COPY --from=build /src/target/release/sessions /app/sessions
ENV DATA_DIR=/data OUTPUT_DIR=/output
CMD ["/app/sessions"]
