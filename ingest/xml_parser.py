import xml.etree.ElementTree as ET
import pandas as pd


def parse_nbi_xml(xml_path: str) -> pd.DataFrame:
    """
    Parses NBI data from an XML file.
    Assumes a structure where bridges are represented as parent nodes containing child tags
    corresponding to NBI fields, e.g.:
    <Bridges>
        <Bridge>
            <STATE_CODE_001>01</STATE_CODE_001>
            <STRUCTURE_NUMBER_008>000001</STRUCTURE_NUMBER_008>
            ...
        </Bridge>
    </Bridges>
    """
    print(f"[INFO] Parsing XML file: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the tag that represents a single bridge record
    # Handles dynamic roots by checking children
    records = []
    
    # We look for nodes that have children (representing key-value attributes of the bridge)
    # The XML tag for a bridge might be 'Bridge' or 'record' or similar
    for child in root:
        record = {}
        for attribute in child:
            record[attribute.tag] = attribute.text
        if record:
            records.append(record)
            
    if not records:
        # Try finding all elements recursively if the root isn't a direct list
        # Look for any elements that have at least one child but no grandchildren
        for node in root.iter():
            if len(node) > 0 and all(len(c) == 0 for c in node):
                record = {c.tag: c.text for c in node}
                records.append(record)

    df = pd.DataFrame(records)
    print(f"[OK] Parsed XML: {len(df)} records found")
    return df
