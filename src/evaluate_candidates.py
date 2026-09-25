import os
import sys
import time
import json
import duckdb
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import normalization as norm

def main():
    start_time = time.time()
    print("=" * 60)
    print("STEP 3C: CANDIDATE GENERATION & RECALL EVALUATION")
    print("=" * 60)
    
    output_dir = "student_resource/outputs/step3_baseline/reports"
    os.makedirs(output_dir, exist_ok=True)
    
    con = duckdb.connect()
    con.execute("SET max_memory = '3.5GB'")
    con.execute("SET threads = 4")
    
    # Register UDFs in DuckDB for fast vectorized preprocessing
    print("[1/5] Registering normalization functions in DuckDB...")
    con.create_function("clean_name", norm.clean_business_name, ["VARCHAR"], "VARCHAR")
    con.create_function("root_name", lambda x: norm.get_root_name(norm.clean_business_name(x)), ["VARCHAR"], "VARCHAR")
    con.create_function("clean_addr", norm.clean_business_address, ["VARCHAR"], "VARCHAR")
    con.create_function("first_num", lambda x: norm.extract_first_num_token(norm.clean_business_address(x)), ["VARCHAR"], "VARCHAR")
    con.create_function("first_2_tok", lambda x: norm.get_first_n_tokens(norm.get_root_name(norm.clean_business_name(x)), 2), ["VARCHAR"], "VARCHAR")
    con.create_function("first_6_root", lambda x: norm.clean_business_name(norm.get_root_name(norm.clean_business_name(x)))[:6], ["VARCHAR"], "VARCHAR")
    con.create_function("region_3", norm.extract_region_prefix, ["VARCHAR"], "VARCHAR")
    con.create_function("first_3_name", lambda x: norm.clean_business_name(x)[:3], ["VARCHAR"], "VARCHAR")
    con.create_function("norm_country", norm.normalize_text, ["VARCHAR"], "VARCHAR")
    
    # Take a stratified evaluation sample of S1 entities to evaluate recall efficiently and safely
    # 50,000 S1 entities across US and India (gives ~173,000 ground truth links, margin of error < 0.1%)
    SAMPLE_SIZE = 50000
    print(f"[2/5] Sampling {SAMPLE_SIZE} S1 evaluation reference entities...")
    
    con.execute(f"""
        CREATE TABLE s1_eval AS 
        SELECT 
            entity_id as s1_id,
            norm_country(country) as country,
            clean_name(business_name) as c_name,
            root_name(business_name) as r_name,
            clean_addr(business_address) as c_addr,
            business_address as raw_addr,
            first_num(business_address) as num_token,
            first_2_tok(business_name) as name_2tok,
            first_6_root(business_name) as name_6char,
            region_3(business_address) as reg_3char,
            first_3_name(business_name) as name_3char
        FROM read_csv('student_resource/dataset/train/train_source1.tsv', delim='\t', header=true, all_varchar=true)
        USING SAMPLE {SAMPLE_SIZE} (reservoir, 42);
    """)
    
    # Build Ground Truth mappings for the evaluation sample
    print("[3/5] Loading ground-truth matches for evaluation sample...")
    con.execute("""
        CREATE TABLE gt_eval AS
        WITH unnested AS (
            SELECT 
                source1_entity_id as s1_id,
                UNNEST(string_split(matched_entity_ids, ',')) as matched_id
            FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true, all_varchar=true)
            WHERE source1_entity_id IN (SELECT s1_id FROM s1_eval)
              AND matched_entity_ids IS NOT NULL 
              AND TRIM(matched_entity_ids) != ''
        )
        SELECT s1_id, matched_id, 
               CASE WHEN matched_id LIKE 'S2-%' THEN 2 ELSE 3 END as source_type
        FROM unnested;
    """)
    
    total_eval_positives = con.execute("SELECT COUNT(*) FROM gt_eval").fetchone()[0]
    total_eval_s2_pos = con.execute("SELECT COUNT(*) FROM gt_eval WHERE source_type = 2").fetchone()[0]
    total_eval_s3_pos = con.execute("SELECT COUNT(*) FROM gt_eval WHERE source_type = 3").fetchone()[0]
    print(f"       Evaluation ground-truth positives: {total_eval_positives:,} ({total_eval_s2_pos:,} S2, {total_eval_s3_pos:,} S3)")
    
    # Process S2 candidates
    print("[4/5] Precomputing blocking keys and generating candidates from Source 2 and Source 3...")
    
    # Generate S1 keys
    con.execute("""
        CREATE TABLE s1_keys AS
        SELECT 
            s1_id, country, c_name, c_addr,
            r_name || '||' || country as k1,
            CASE WHEN name_2tok != '' AND num_token != '' THEN name_2tok || '||' || num_token || '||' || country ELSE NULL END as k2,
            CASE WHEN LEN(name_6char) >= 3 AND reg_3char != '' THEN name_6char || '||' || reg_3char || '||' || country ELSE NULL END as k3,
            CASE WHEN num_token != '' AND LEN(name_3char) >= 2 THEN num_token || '||' || name_3char || '||' || country ELSE NULL END as k4
        FROM s1_eval;
    """)
    
    # Process S2
    print("      Indexing Source 2 (5.0M records)...")
    con.execute("""
        CREATE TEMP TABLE s2_prep AS
        SELECT 
            entity_id as cand_id,
            clean_name(business_name) as c_name,
            clean_addr(business_address) as c_addr,
            root_name(business_name) || '||' || norm_country(country) as k1,
            CASE WHEN first_2_tok(business_name) != '' AND first_num(business_address) != '' 
                 THEN first_2_tok(business_name) || '||' || first_num(business_address) || '||' || norm_country(country) 
                 ELSE NULL END as k2,
            CASE WHEN LEN(first_6_root(business_name)) >= 3 AND region_3(business_address) != '' 
                 THEN first_6_root(business_name) || '||' || region_3(business_address) || '||' || norm_country(country) 
                 ELSE NULL END as k3,
            CASE WHEN first_num(business_address) != '' AND LEN(first_3_name(business_name)) >= 2 
                 THEN first_num(business_address) || '||' || first_3_name(business_name) || '||' || norm_country(country) 
                 ELSE NULL END as k4
        FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true, all_varchar=true);
    """)
    
    # Prune keys with > 150 records in S2
    print("      Pruning high-frequency blocks (>150 records) in S2...")
    con.execute("CREATE TEMP TABLE s2_k1_valid AS SELECT k1 FROM s2_prep WHERE k1 IS NOT NULL GROUP BY k1 HAVING COUNT(*) <= 150;")
    con.execute("CREATE TEMP TABLE s2_k2_valid AS SELECT k2 FROM s2_prep WHERE k2 IS NOT NULL GROUP BY k2 HAVING COUNT(*) <= 150;")
    con.execute("CREATE TEMP TABLE s2_k3_valid AS SELECT k3 FROM s2_prep WHERE k3 IS NOT NULL GROUP BY k3 HAVING COUNT(*) <= 150;")
    con.execute("CREATE TEMP TABLE s2_k4_valid AS SELECT k4 FROM s2_prep WHERE k4 IS NOT NULL GROUP BY k4 HAVING COUNT(*) <= 150;")
    
    # Match S1 against S2
    print("      Matching S1 evaluation entities against S2...")
    con.execute("""
        CREATE TEMP TABLE s2_candidates AS
        WITH matches AS (
            SELECT s1.s1_id, s2.cand_id, 1 as pass_id
            FROM s1_keys s1 JOIN s2_prep s2 ON s1.k1 = s2.k1
            JOIN s2_k1_valid v ON s1.k1 = v.k1
            UNION ALL
            SELECT s1.s1_id, s2.cand_id, 2 as pass_id
            FROM s1_keys s1 JOIN s2_prep s2 ON s1.k2 = s2.k2
            JOIN s2_k2_valid v ON s1.k2 = v.k2
            UNION ALL
            SELECT s1.s1_id, s2.cand_id, 3 as pass_id
            FROM s1_keys s1 JOIN s2_prep s2 ON s1.k3 = s2.k3
            JOIN s2_k3_valid v ON s1.k3 = v.k3
            UNION ALL
            SELECT s1.s1_id, s2.cand_id, 4 as pass_id
            FROM s1_keys s1 JOIN s2_prep s2 ON s1.k4 = s2.k4
            JOIN s2_k4_valid v ON s1.k4 = v.k4
        )
        SELECT s1_id, cand_id, COUNT(DISTINCT pass_id) as passes_matched, MIN(pass_id) as min_pass_id
        FROM matches
        GROUP BY s1_id, cand_id;
    """)
    con.execute("DROP TABLE s2_prep; DROP TABLE s2_k1_valid; DROP TABLE s2_k2_valid; DROP TABLE s2_k3_valid; DROP TABLE s2_k4_valid;")
    
    # Process S3
    print("      Indexing Source 3 (5.3M records)...")
    con.execute("""
        CREATE TEMP TABLE s3_prep AS
        SELECT 
            entity_id as cand_id,
            clean_name(business_name) as c_name,
            clean_addr(business_address) as c_addr,
            root_name(business_name) || '||' || norm_country(country) as k1,
            CASE WHEN first_2_tok(business_name) != '' AND first_num(business_address) != '' 
                 THEN first_2_tok(business_name) || '||' || first_num(business_address) || '||' || norm_country(country) 
                 ELSE NULL END as k2,
            CASE WHEN LEN(first_6_root(business_name)) >= 3 AND region_3(business_address) != '' 
                 THEN first_6_root(business_name) || '||' || region_3(business_address) || '||' || norm_country(country) 
                 ELSE NULL END as k3,
            CASE WHEN first_num(business_address) != '' AND LEN(first_3_name(business_name)) >= 2 
                 THEN first_num(business_address) || '||' || first_3_name(business_name) || '||' || norm_country(country) 
                 ELSE NULL END as k4
        FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true, all_varchar=true);
    """)
    
    # Prune keys with > 150 records in S3
    print("      Pruning high-frequency blocks (>150 records) in S3...")
    con.execute("CREATE TEMP TABLE s3_k1_valid AS SELECT k1 FROM s3_prep WHERE k1 IS NOT NULL GROUP BY k1 HAVING COUNT(*) <= 150;")
    con.execute("CREATE TEMP TABLE s3_k2_valid AS SELECT k2 FROM s3_prep WHERE k2 IS NOT NULL GROUP BY k2 HAVING COUNT(*) <= 150;")
    con.execute("CREATE TEMP TABLE s3_k3_valid AS SELECT k3 FROM s3_prep WHERE k3 IS NOT NULL GROUP BY k3 HAVING COUNT(*) <= 150;")
    con.execute("CREATE TEMP TABLE s3_k4_valid AS SELECT k4 FROM s3_prep WHERE k4 IS NOT NULL GROUP BY k4 HAVING COUNT(*) <= 150;")
    
    # Match S1 against S3
    print("      Matching S1 evaluation entities against S3...")
    con.execute("""
        CREATE TEMP TABLE s3_candidates AS
        WITH matches AS (
            SELECT s1.s1_id, s3.cand_id, 1 as pass_id
            FROM s1_keys s1 JOIN s3_prep s3 ON s1.k1 = s3.k1
            JOIN s3_k1_valid v ON s1.k1 = v.k1
            UNION ALL
            SELECT s1.s1_id, s3.cand_id, 2 as pass_id
            FROM s1_keys s1 JOIN s3_prep s3 ON s1.k2 = s3.k2
            JOIN s3_k2_valid v ON s1.k2 = v.k2
            UNION ALL
            SELECT s1.s1_id, s3.cand_id, 3 as pass_id
            FROM s1_keys s1 JOIN s3_prep s3 ON s1.k3 = s3.k3
            JOIN s3_k3_valid v ON s1.k3 = v.k3
            UNION ALL
            SELECT s1.s1_id, s3.cand_id, 4 as pass_id
            FROM s1_keys s1 JOIN s3_prep s3 ON s1.k4 = s3.k4
            JOIN s3_k4_valid v ON s1.k4 = v.k4
        )
        SELECT s1_id, cand_id, COUNT(DISTINCT pass_id) as passes_matched, MIN(pass_id) as min_pass_id
        FROM matches
        GROUP BY s1_id, cand_id;
    """)
    con.execute("DROP TABLE s3_prep; DROP TABLE s3_k1_valid; DROP TABLE s3_k2_valid; DROP TABLE s3_k3_valid; DROP TABLE s3_k4_valid;")
    
    # Union S2 and S3 candidates
    con.execute("""
        CREATE TABLE all_candidates AS
        SELECT s1_id, cand_id, passes_matched, min_pass_id, 2 as source_type FROM s2_candidates
        UNION ALL
        SELECT s1_id, cand_id, passes_matched, min_pass_id, 3 as source_type FROM s3_candidates;
    """)
    
    total_candidates_found = con.execute("SELECT COUNT(*) FROM all_candidates").fetchone()[0]
    avg_cands_per_s1 = round(total_candidates_found / SAMPLE_SIZE, 2)
    print(f"      Total candidates generated: {total_candidates_found:,} (Average: {avg_cands_per_s1} per S1 entity)")
    
    # Calculate recall across K = 10, 15, 20, 25
    print("\n[5/5] Calculating candidate recall against ground truth for K in {10, 15, 20, 25}...")
    
    # Rank candidates per S1 entity by (passes_matched DESC, min_pass_id ASC)
    con.execute("""
        CREATE TABLE ranked_candidates AS
        SELECT 
            s1_id, cand_id, source_type, passes_matched, min_pass_id,
            ROW_NUMBER() OVER (PARTITION BY s1_id, source_type ORDER BY passes_matched DESC, min_pass_id ASC) as rank_within_source,
            ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY passes_matched DESC, min_pass_id ASC) as rank_overall
        FROM all_candidates;
    """)
    
    k_values = [10, 15, 20, 25]
    recall_results = {}
    
    for k in k_values:
        # Candidate retained if rank_overall <= k
        con.execute(f"""
            CREATE TEMP TABLE recalled_k AS
            SELECT g.s1_id, g.matched_id, g.source_type
            FROM gt_eval g
            JOIN ranked_candidates c ON g.s1_id = c.s1_id AND g.matched_id = c.cand_id
            WHERE c.rank_overall <= {k};
        """)
        recalled_total = con.execute("SELECT COUNT(*) FROM recalled_k").fetchone()[0]
        recalled_s2 = con.execute("SELECT COUNT(*) FROM recalled_k WHERE source_type = 2").fetchone()[0]
        recalled_s3 = con.execute("SELECT COUNT(*) FROM recalled_k WHERE source_type = 3").fetchone()[0]
        
        recall_pct = round(100.0 * recalled_total / total_eval_positives, 4)
        recall_s2_pct = round(100.0 * recalled_s2 / total_eval_s2_pos, 4)
        recall_s3_pct = round(100.0 * recalled_s3 / total_eval_s3_pos, 4)
        
        recall_results[f"K={k}"] = {
            "k": k,
            "overall_recall_pct": recall_pct,
            "recalled_total": recalled_total,
            "total_positives": total_eval_positives,
            "s2_recall_pct": recall_s2_pct,
            "s3_recall_pct": recall_s3_pct
        }
        print(f"      >> K = {k:2d}: Overall Recall = {recall_pct:6.2f}% | S2 Recall = {recall_s2_pct:6.2f}% | S3 Recall = {recall_s3_pct:6.2f}%")
        con.execute("DROP TABLE recalled_k;")
    
    elapsed = round(time.time() - start_time, 2)
    print(f"\nExecution completed in {elapsed} seconds.")
    
    report_file = os.path.join(output_dir, "candidate_recall_report.json")
    with open(report_file, "w") as fp:
        json.dump({
            "sample_size": SAMPLE_SIZE,
            "total_eval_positives": total_eval_positives,
            "avg_candidates_per_s1": avg_cands_per_s1,
            "recall_by_k": recall_results,
            "elapsed_seconds": elapsed
        }, fp, indent=2)
    print(f"Report saved to: {report_file}")

if __name__ == "__main__":
    main()
