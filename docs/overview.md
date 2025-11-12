# Docs — compress-photos-cli
Additional notes and flow diagrams.
## Processing pipeline
```mermaid
flowchart LR
  CLI[CLI] -->|read| Loader[Loader]
  Loader[Loader] -->|resize/encode| Processor[Processor]
  Processor[Processor] -->|save| Writer[Writer]
```
