# Changelog

## 1.0.0 — release candidate

- Retained reviewer source modules and synthetic tests.
- Ported documentary ablation and temporal analyses from final reviewer notebook.
- Removed redundant local metrics imports that shadowed the module-level import in temporal validation.
- Explicitly pass the configured bootstrap count to the primary analysis (2000 in final mode), avoiding a stale Python default argument.
- Updated authorship, article citation, MIT license and documentation.
