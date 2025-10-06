import json
import subprocess


def diagnose_ast_issue(source_file: str, ast_file: str):
    """Comprehensive diagnostic for AST parsing issues"""
    
    # Load files
    with open(source_file, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    with open(ast_file, 'r', encoding='utf-8') as f:
        ast_data = json.load(f)
    
    print("=" * 80)
    print("AST PARSING DIAGNOSTIC")
    print("=" * 80)
    
    # Check 1: File encoding and special characters
    print("\n📄 FILE ANALYSIS:")
    print(f"   Source code length: {len(source_code)} characters")
    print(f"   Source code lines: {len(source_code.splitlines())}")
    print(f"   First 10 chars (repr): {repr(source_code[:10])}")
    print(f"   Last 10 chars (repr): {repr(source_code[-10:])}")
    
    # Check for BOM
    if source_code.startswith('\ufeff'):
        print("   ⚠️  WARNING: File has UTF-8 BOM (Byte Order Mark)")
        print("   This can cause index misalignment!")
        source_code = source_code.lstrip('\ufeff')
        print("   → Stripped BOM for analysis")
    
    # Check line endings
    has_crlf = '\r\n' in source_code
    has_lf = '\n' in source_code and not has_crlf
    line_ending_type = 'CRLF (\\r\\n)' if has_crlf else 'LF (\\n)' if has_lf else 'Unknown'
    print(f"   Line endings: {line_ending_type}")
    
    # Check 2: AST root level
    print("\n🌳 AST ROOT ANALYSIS:")
    print(f"   AST type: {ast_data.get('type')}")
    print(f"   AST start: {ast_data.get('start')}")
    print(f"   AST end: {ast_data.get('end')}")
    print(f"   Source length: {len(source_code)}")
    
    if ast_data.get('end') != len(source_code):
        diff = ast_data.get('end') - len(source_code)
        print(f"   ⚠️  MISMATCH: AST end is {diff:+d} characters off!")
    else:
        print("   ✓ AST end matches source length")
    
    # Check 3: First few imports
    print("\n📦 IMPORT STATEMENT ANALYSIS:")
    
    body = ast_data.get('program', {}).get('body', [])
    imports = [node for node in body if node.get('type') == 'ImportDeclaration']
    
    print(f"   Found {len(imports)} import statements")
    
    for i, imp in enumerate(imports[:5]):  # Check first 5
        start = imp['start']
        end = imp['end']
        source_val = imp.get('source', {}).get('value', 'N/A')
        
        print(f"\n   Import #{i+1}: from '{source_val}'")
        print(f"   → AST indices: [{start}:{end}] (length: {end-start})")
        
        # Extract using AST indices
        extracted = source_code[start:end]
        print(f"   → Extracted: '{extracted}'")
        
        # Check if it looks correct
        if extracted.strip().startswith('import'):
            print(f"   ✓ Extraction looks correct")
        else:
            print(f"   ✗ Extraction WRONG! Should start with 'import'")
            
            # Try to find the actual import
            search_str = f'from "{source_val}"'
            actual_pos = source_code.find(search_str)
            
            if actual_pos != -1:
                # Find the full line
                line_start = source_code.rfind('\n', 0, actual_pos)
                line_start = line_start + 1 if line_start != -1 else 0
                
                line_end = source_code.find('\n', actual_pos)
                line_end = line_end if line_end != -1 else len(source_code)
                
                actual_import = source_code[line_start:line_end]
                
                print(f"   → Found actual import at position {actual_pos}")
                print(f"   → Actual line indices: [{line_start}:{line_end}]")
                print(f"   → Actual import: '{actual_import}'")
                print(f"   → Offset: AST is {start - line_start} characters off")
    
    # Check 4: Re-parse to verify
    print("\n🔄 RE-PARSING TEST:")
    try:
        result = subprocess.run(
            ["node", "ast_generator.js"],
            input=source_code,
            capture_output=True,
            text=True,
            check=False,
        )
        
        if result.returncode != 0:
            print(f"   ✗ Parser returned error: {result.stderr}")
        else:
            # Check output format
            output = result.stdout
            print(f"   Output length: {len(output)} characters")
            print(f"   Starts with '{{'? {output.strip().startswith('{')}")
            
            # Try to parse
            try:
                fresh_ast = json.loads(output)
                
                # Compare with saved AST
                fresh_end = fresh_ast.get('end')
                saved_end = ast_data.get('end')
                
                print(f"   Fresh parse end: {fresh_end}")
                print(f"   Saved AST end: {saved_end}")
                
                if fresh_end != saved_end:
                    print(f"   ⚠️  ASTs differ! Saved AST may be outdated or corrupted")
                    print(f"   → Recommend regenerating AST file")
                else:
                    print(f"   ✓ Fresh parse matches saved AST")
                
                # Check if fresh parse is correct
                if fresh_end == len(source_code):
                    print(f"   ✓ Fresh parse has correct indices")
                else:
                    print(f"   ✗ Fresh parse also has wrong indices!")
                    print(f"   → Issue is in ast_generator.js or source file encoding")
                    
            except json.JSONDecodeError as e:
                print(f"   ✗ Fresh parse output is not valid JSON: {e}")
                print(f"   First 200 chars: {repr(output[:200])}")
                
    except FileNotFoundError:
        print("   ✗ Could not run ast_generator.js (Node.js not found?)")
    except Exception as e:
        print(f"   ✗ Error during re-parse: {e}")
    
    # Check 5: Source code integrity
    print("\n🔍 SOURCE CODE INTEGRITY:")
    
    # Check for null bytes or other weird characters
    if '\x00' in source_code:
        print("   ⚠️  WARNING: Source contains null bytes!")
    
    # Check first import manually
    first_import_line = None
    for line in source_code.split('\n'):
        if line.strip().startswith('import'):
            first_import_line = line
            break
    
    if first_import_line:
        print(f"   First import line: '{first_import_line}'")
        print(f"   Position in source: {source_code.find(first_import_line)}")
        
        if imports:
            first_ast_import = imports[0]
            ast_start = first_ast_import['start']
            print(f"   AST says it starts at: {ast_start}")
            print(f"   Difference: {source_code.find(first_import_line) - ast_start}")
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    
    issues = []
    
    if ast_data.get('end') != len(source_code):
        issues.append("AST end position doesn't match source code length")
    
    if any(not source_code[imp['start']:imp['end']].strip().startswith('import') 
           for imp in imports[:3]):
        issues.append("Import extractions are misaligned")
    
    if source_code.startswith('\ufeff'):
        issues.append("Source file has BOM (Byte Order Mark)")
    
    if issues:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   • {issue}")
        
        print("\n💡 RECOMMENDED FIXES:")
        
        if "BOM" in str(issues):
            print("   1. Remove BOM from source file:")
            print("      with open('file.jsx', 'r', encoding='utf-8-sig') as f:")
            print("          content = f.read()")
            print("      with open('file.jsx', 'w', encoding='utf-8') as f:")
            print("          f.write(content)")
        
        if "AST end" in str(issues) or "misaligned" in str(issues):
            print("   2. Regenerate AST file:")
            print("      python your_parsing_script.py")
            print("   3. Ensure ast_generator.js outputs ONLY JSON")
            print("   4. Check that source file encoding is consistent")
    else:
        print("✓ No issues found! AST looks correct.")
        print("\nIf you're still seeing problems, the issue might be in how you're")
        print("using the CodeChunker class. Try running it with verbose output.")


# Quick fix function
def fix_and_regenerate(source_file: str):
    """Clean source file and regenerate AST"""
    print("\n🔧 ATTEMPTING TO FIX...")
    
    # Read with BOM handling
    with open(source_file, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    # Write back without BOM
    cleaned_file = source_file.replace('.jsx', '_cleaned.jsx')
    with open(cleaned_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"   ✓ Created cleaned file: {cleaned_file}")
    
    # Regenerate AST
    try:
        result = subprocess.run(
            ["node", "ast_generator.js"],
            input=content,
            capture_output=True,
            text=True,
            check=True,
        )
        
        ast_data = json.loads(result.stdout)
        
        ast_file = cleaned_file.replace('.jsx', '_ast.json')
        with open(ast_file, 'w', encoding='utf-8') as f:
            json.dump(ast_data, f, indent=2)
        
        print(f"   ✓ Generated new AST: {ast_file}")
        print(f"\n   Try using these files:")
        print(f"   - Source: {cleaned_file}")
        print(f"   - AST: {ast_file}")
        
        return cleaned_file, ast_file
        
    except Exception as e:
        print(f"   ✗ Failed to regenerate: {e}")
        return None, None


if __name__ == "__main__":
    # Run diagnostics
    diagnose_ast_issue("./samples/App.jsx", "astApp.json")
    
    # Offer to fix
    print("\n" + "=" * 80)
    response = input("Would you like to attempt automatic fix? (y/n): ")
    
    if response.lower() == 'y':
        fix_and_regenerate("./samples/App.jsx")