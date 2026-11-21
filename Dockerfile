# Shopping Sessions: Rust pipeline.
FROM rust:1-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY .cargo ./.cargo
# Cache dependencies in their own layer.
RUN mkdir src && echo "fn main(){}" > src/main.rs && cargo build --release && rm -rf src
COPY src ./src
RUN touch src/main.rs && cargo build --release

FROM debian:bookworm-slim
COPY --from=build /src/target/release/sessions /app/sessions
ENV DATA_DIR=/data OUTPUT_DIR=/output
CMD ["/app/sessions"]
