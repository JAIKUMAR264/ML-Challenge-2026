import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import normalization as norm

class TestNormalizationPipeline(unittest.TestCase):

    # 1. Unicode and case normalization
    def test_unicode_and_case(self):
        self.assertEqual(norm.normalize_text("  ACME   CORP.  "), "acme corp.")
        self.assertEqual(norm.normalize_text("Café & Résumé"), "cafe and resume")
        self.assertEqual(norm.normalize_text("“Smart” Solutions ‘LLC’"), '"smart" solutions \'llc\'')
        self.assertEqual(norm.normalize_text("Floor-5"), "floor-5")
        self.assertEqual(norm.clean_business_name("HÔTEL   RÀM & CÔ.!"), "hotel ram and co")

    # 2. Honorific prefixes
    def test_honorific_prefixes(self):
        # Leading honorifics should be removed
        self.assertEqual(norm.strip_honorifics("sri iri enterprise"), "iri enterprise")
        self.assertEqual(norm.strip_honorifics("shri ultra solutions"), "ultra solutions")
        self.assertEqual(norm.strip_honorifics("smt kamala trading"), "kamala trading")
        self.assertEqual(norm.strip_honorifics("m/s balaji steels"), "balaji steels")
        self.assertEqual(norm.strip_honorifics("dr smart marketing"), "smart marketing")
        self.assertEqual(norm.strip_honorifics("messrs smith and co"), "smith and co")
        
        # Internal word boundary preservation: honorific prefix should NOT strip from real names!
        self.assertEqual(norm.strip_honorifics("srinivasan technologies"), "srinivasan technologies")
        self.assertEqual(norm.strip_honorifics("shrikant textiles"), "shrikant textiles")
        self.assertEqual(norm.strip_honorifics("sriram enterprises"), "sriram enterprises")

    # 3. Legal suffixes (US, India, France)
    def test_legal_suffixes(self):
        # India
        self.assertEqual(norm.strip_legal_suffixes("tata consultancy private limited"), "tata consultancy")
        self.assertEqual(norm.strip_legal_suffixes("infosys pvt ltd"), "infosys")
        self.assertEqual(norm.strip_legal_suffixes("digital energy limited private"), "digital energy")
        self.assertEqual(norm.strip_legal_suffixes("yf solutions [private]"), "yf solutions")
        
        # US
        self.assertEqual(norm.strip_legal_suffixes("cardiology care associates of gilbert llc"), "cardiology care associates of gilbert")
        self.assertEqual(norm.strip_legal_suffixes("apple incorporated"), "apple")
        self.assertEqual(norm.strip_legal_suffixes("microsoft corp"), "microsoft")
        self.assertEqual(norm.strip_legal_suffixes("pegeen anzures superior sunrise (llc)"), "pegeen anzures superior sunrise")
        self.assertEqual(norm.strip_legal_suffixes("huber continental newbury llp"), "huber continental newbury")

        # France
        self.assertEqual(norm.strip_legal_suffixes("znb club sarl"), "znb club")
        self.assertEqual(norm.strip_legal_suffixes("thermal and fils sasu"), "thermal")
        self.assertEqual(norm.strip_legal_suffixes("elephant centre eurl"), "elephant centre")
        self.assertEqual(norm.strip_legal_suffixes("grain and fils"), "grain")
        self.assertEqual(norm.strip_legal_suffixes("dupond selarl"), "dupond")

    # 4. Reversed name and address components
    def test_reversed_components(self):
        # Name token order variation
        sorted_1 = norm.get_sorted_tokens(norm.get_root_name(norm.clean_business_name("Innovative Research Systems")))
        sorted_2 = norm.get_sorted_tokens(norm.get_root_name(norm.clean_business_name("RESEARCH SYSTEMS INNOVATIVE")))
        self.assertEqual(sorted_1, sorted_2)
        self.assertEqual(sorted_1, "innovative research")

        # Reversed address components (state at front vs state at end)
        s1_addr = "163 B Old Mill Road, High Point, NC"
        s2_addr = "NC, 163 B OLD MILL ROAD, HIGH POINT"
        self.assertEqual(norm.standardize_region(s1_addr, "US"), "nc")
        self.assertEqual(norm.standardize_region(s2_addr, "US"), "nc")

        # India state reversed
        s1_in = "4/7 A J C Bose Road, Kolkata, Calcutta, West Bengal"
        s2_in = "West Bengal, 4/7 A J C BOSE ROAD, KOLKATA"
        self.assertEqual(norm.standardize_region(s1_in, "India"), "wb")
        self.assertEqual(norm.standardize_region(s2_in, "India"), "wb")

    # 5. Missing names and addresses
    def test_missing_values(self):
        self.assertTrue(norm.is_missing_value(None))
        self.assertTrue(norm.is_missing_value(""))
        self.assertTrue(norm.is_missing_value("None"))
        self.assertTrue(norm.is_missing_value("null"))
        self.assertTrue(norm.is_missing_value("NULL"))
        self.assertTrue(norm.is_missing_value("N/A"))
        self.assertTrue(norm.is_missing_value("nan"))
        self.assertTrue(norm.is_missing_value("NaN"))
        self.assertTrue(norm.is_missing_value("NA"))

        self.assertEqual(norm.clean_business_name("None"), "")
        self.assertEqual(norm.clean_business_name("N/A"), "")
        self.assertEqual(norm.clean_business_name("NA"), "")
        self.assertEqual(norm.clean_business_address("None"), "")
        self.assertEqual(norm.clean_business_address("null"), "")
        self.assertEqual(norm.normalize_country(None), "")

    # 6. Multilingual text
    def test_multilingual_text(self):
        # Hindi
        hindi = "होटल राम कंसल्टिंग प्राइवेट लिमिटेड"
        clean_hi = norm.clean_business_name(hindi)
        self.assertTrue(len(clean_hi) > 0)
        self.assertIn("होटल", clean_hi)
        self.assertIn("राम", clean_hi)

        # Gujarati
        guj = "પાયોનિયર સોફ્ટવેર"
        clean_guj = norm.clean_business_name(guj)
        self.assertTrue(len(clean_guj) > 0)
        self.assertIn("પાયોનિયર", clean_guj)

        # Bengali
        bengali = "স্মার্ট মার্কেটিং প্রাইভেট লিমিটেড"
        clean_bn = norm.clean_business_name(bengali)
        self.assertTrue(len(clean_bn) > 0)
        self.assertIn("স্মার্ট", clean_bn)

        # French diacritics
        fr = "Hôtel & Crêperie de l'Étoile"
        clean_fr = norm.clean_business_name(fr)
        self.assertEqual(clean_fr, "hotel and creperie de l etoile")

    # 7. Numeric address extraction
    def test_numeric_extraction(self):
        self.assertEqual(norm.extract_numeric_tokens("163 B Old Mill Road"), ["163"])
        self.assertEqual(norm.extract_first_num_token("163 B Old Mill Road"), "163")
        self.assertEqual(norm.extract_first_num_token("Plot No. 626/2, Phase 4"), "626/2")
        self.assertEqual(norm.extract_first_num_token("Cs No. 5 D/1, 5 B/6"), "5")
        self.assertEqual(norm.extract_first_num_token("Door No 183, 41St Cross"), "183")
        self.assertEqual(norm.extract_first_num_token("No Address Provided"), "")

    # 8. Country normalization, including France
    def test_country_normalization(self):
        self.assertEqual(norm.normalize_country("US"), "us")
        self.assertEqual(norm.normalize_country("usa"), "us")
        self.assertEqual(norm.normalize_country("United States"), "us")
        self.assertEqual(norm.normalize_country("India"), "india")
        self.assertEqual(norm.normalize_country("ind"), "india")
        self.assertEqual(norm.normalize_country("France"), "france")
        self.assertEqual(norm.normalize_country("fr"), "france")
        self.assertEqual(norm.normalize_country("FRA"), "france")
        # Unseen countries must NOT be rejected or dropped
        self.assertEqual(norm.normalize_country("Germany"), "germany")
        self.assertEqual(norm.normalize_country("Brazil"), "brazil")

    # 9. Preservation of entity IDs and raw fields
    def test_entity_id_preservation(self):
        raw_record = {
            "entity_id": "S1-99887766",
            "business_name": "Acme Inc.",
            "business_address": "123 Main St, Austin, Texas",
            "country": "US"
        }
        # Verify normalization does NOT alter raw fields
        c_name = norm.clean_business_name(raw_record["business_name"])
        r_name = norm.get_root_name(c_name)
        c_addr = norm.clean_business_address(raw_record["business_address"])
        c_country = norm.normalize_country(raw_record["country"])

        self.assertEqual(raw_record["entity_id"], "S1-99887766")
        self.assertEqual(raw_record["business_name"], "Acme Inc.")
        self.assertEqual(r_name, "acme")
        self.assertEqual(c_country, "us")

    # 10. Consistent behavior across all three sources
    def test_cross_source_consistency(self):
        s1 = {"name": "Iri Enterprise", "addr": "P 21 South Extention, Delhi", "c": "India"}
        s2 = {"name": "Sri Iri Enterprise", "addr": "#161 P 21 SOUTH EXTENTION, Delhi", "c": "India"}
        s3 = {"name": "Iri Enterprise Pvt Ltd", "addr": "21 P South Extention, Delhi", "c": "India"}

        r1 = norm.get_root_name(norm.clean_business_name(s1["name"]))
        r2 = norm.get_root_name(norm.clean_business_name(s2["name"]))
        r3 = norm.get_root_name(norm.clean_business_name(s3["name"]))

        self.assertEqual(r1, "iri enterprise")
        self.assertEqual(r2, "iri enterprise")
        self.assertEqual(r3, "iri enterprise")
        self.assertEqual(r1, r2)
        self.assertEqual(r1, r3)

if __name__ == '__main__':
    unittest.main()
