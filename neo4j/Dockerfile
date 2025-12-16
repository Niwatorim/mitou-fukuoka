FROM neo4j:5

# Install APOC plugin
ENV NEO4J_PLUGINS='["apoc"]'

# Allow APOC export/import
ENV NEO4J_apoc_export_file_enabled=true
ENV NEO4J_apoc_import_file_enabled=true
ENV NEO4J_apoc_import_file_use__neo4j__config=true
ENV NEO4J_dbms_security_procedures_unrestricted=apoc.*
