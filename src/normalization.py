import re
import unicodedata

# Legal / corporate entity suffixes to strip for root_name
LEGAL_SUFFIXES_PATTERN = re.compile(
    r'\b(?:private limited|pvt ltd|incorporated|corporation|company|limited|pvt|corp|inc|llc|ltd|co|sarl|sas|eurl|sa|llp|gmbh|spa|bv)\b',
    re.IGNORECASE
)

# Numeric address token pattern (e.g., 570/13, 201/D, 113/154, 105, 3907)
NUMERIC_ADDR_PATTERN = re.compile(r'\b\d+(?:[/-]\d+|[a-zA-Z])?\b')

def normalize_text(text: str) -> str:
    """Apply Unicode NFKD normalization, strip combining diacritics, and lowercase."""
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip()

def clean_business_name(text: str) -> str:
    """Normalize business name: lowercase, punctuation removed, whitespace collapsed."""
    text = normalize_text(text)
    # Replace non-alphanumeric characters with space
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_root_name(clean_name: str) -> str:
    """Remove legal suffixes from clean business name to produce root_name."""
    if not clean_name:
        return ""
    root = LEGAL_SUFFIXES_PATTERN.sub(' ', clean_name)
    root = re.sub(r'\s+', ' ', root).strip()
    return root if root else clean_name

def clean_business_address(text: str) -> str:
    """Normalize address while preserving numeric formats like 570/13, 201/D."""
    text = normalize_text(text)
    if not text:
        return ""
    # Preserve alphanumeric, spaces, and slashes/hyphens used in street numbers
    text = re.sub(r'[^\w\s/-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_numeric_tokens(clean_addr: str) -> list[str]:
    """Extract list of numeric address tokens."""
    if not clean_addr:
        return []
    return NUMERIC_ADDR_PATTERN.findall(clean_addr)

def extract_first_num_token(clean_addr: str) -> str:
    """Return the first numeric address token or empty string."""
    tokens = extract_numeric_tokens(clean_addr)
    return tokens[0] if tokens else ""

def get_first_n_tokens(text: str, n: int = 2) -> str:
    """Extract first n whitespace-delimited tokens."""
    if not text:
        return ""
    tokens = text.split()
    return " ".join(tokens[:n])

def extract_region_prefix(raw_address: str) -> str:
    """Extract 3-char prefix of city/state/region from address."""
    if not raw_address:
        return ""
    # Usually address format is street, city, state
    parts = [p.strip() for p in raw_address.split(',') if p.strip()]
    if len(parts) >= 2:
        # Take the second or last part
        region = normalize_text(parts[-1])
        region = re.sub(r'[^\w]', '', region)
        return region[:3]
    clean = re.sub(r'[^\w]', '', normalize_text(raw_address))
    return clean[-3:] if len(clean) >= 3 else clean

def generate_blocking_keys(root_name: str, clean_name: str, raw_address: str, clean_address: str, country: str) -> dict:
    """
    Generate the 4 blocking keys for candidate generation.
    All keys are strictly suffixed with country to ensure country-aware partitioning.
    """
    c = normalize_text(country)
    first_num = extract_first_num_token(clean_address)
    first_2_tokens = get_first_n_tokens(root_name, 2)
    first_6_root = re.sub(r'[^\w]', '', root_name)[:6]
    region_3 = extract_region_prefix(raw_address)
    first_3_name = re.sub(r'[^\w]', '', clean_name)[:3]

    keys = {
        # Pass 1: exact root_name + country
        "pass1": f"{root_name}||{c}" if root_name else "",
        
        # Pass 2: first 2 significant root-name tokens + first numeric address token + country
        "pass2": f"{first_2_tokens}||{first_num}||{c}" if (first_2_tokens and first_num) else "",
        
        # Pass 3: first 6 alphanumeric chars of root_name + first 3 chars of city/state + country
        "pass3": f"{first_6_root}||{region_3}||{c}" if (len(first_6_root) >= 3 and region_3) else "",
        
        # Pass 4: exact address numeric token + 3-char name prefix + country
        "pass4": f"{first_num}||{first_3_name}||{c}" if (first_num and len(first_3_name) >= 2) else ""
    }
    return keys
