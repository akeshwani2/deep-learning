# Repository Safety Verification

- **Checked:** 2026-10-22
- **Scope:** Git ignore behavior and files tracked at the current revision
- **Result:** Pass

## Ignore-rule assertions

`git check-ignore -v --no-index` confirmed that all 14 representative local
paths are ignored:

- local environment and Synapse configuration;
- credential and secret directories;
- source and derived data;
- participant allocations, split files, and fold files;
- artifacts, uploads, checkpoints, and feature caches; and
- raw audio and serialized model/array formats through their directory or file
  rules.

The check also confirmed that `.env.example` is intentionally trackable. It
contains only empty path-variable names and no credentials or secret values.

## Tracked-file scans

The following checks returned zero findings:

1. A credential-pattern scan over tracked text files for common cloud/API token
   formats, private-key headers, and non-empty key/token/password/secret
   assignments. Generated binary PDF deliverables were excluded from this text
   pattern scan.
2. A tracked-path scan for CSV/TSV files, raw-audio formats, and files below
   restricted data, derived data, participant-allocation, split/fold, artifact,
   upload, checkpoint, credential, or secret directories.

These are repository hygiene checks, not a guarantee that arbitrary future files
are safe. The checks must be rerun before releases and after adding any data or
credential-dependent workflow.
