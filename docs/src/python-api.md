# Python API

The package is organized as a sequence of small, composable modules that mirror
the workflow stages. Each is independently usable and documented below.

## Schema and storage

```{eval-rst}
.. automodule:: src.mock_patient_profile.schema
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.mock_patient_profile.paths
   :members:
   :undoc-members:
   :show-inheritance:
```

## Dataset ingestion and synthesis

```{eval-rst}
.. automodule:: src.mock_patient_profile.bbbc021
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.mock_patient_profile.patients
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.mock_patient_profile.synthetic
   :members:
   :undoc-members:
   :show-inheritance:
```

## Way Science integration layers

```{eval-rst}
.. automodule:: src.mock_patient_profile.cytotable_io
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.mock_patient_profile.cytodataframe_io
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.mock_patient_profile.qc
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.mock_patient_profile.profiling
   :members:
   :undoc-members:
   :show-inheritance:
```

## Multi-omic integration and orchestration

```{eval-rst}
.. automodule:: src.mock_patient_profile.multiomics
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.mock_patient_profile.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.mock_patient_profile.cli
   :members:
   :undoc-members:
   :show-inheritance:
```
