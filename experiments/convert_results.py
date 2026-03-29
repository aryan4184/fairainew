import json
import os

def main():
    json_path = "../Results/validation/validation_results.json"
    txt_path = "../Results/validation/validation_results.txt"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    lines = []
    lines.append("==================================================")
    lines.append("       EMPIRICAL VALIDATION RESULTS REPORT        ")
    lines.append("==================================================")
    lines.append("")
    
    # Group 1 & 2
    if "group_1_and_2" in data:
        lines.append("--------------------------------------------------")
        lines.append("TEST GROUP 1 & 2: BOUND CHECK & BOOTSTRAP")
        lines.append("--------------------------------------------------")
        lines.append(f"{'Mode':<15} | {'Type':<20} | {'LHS (Gap)':<10} | {'RHS (Bound)':<12} | {'Tightness':<10} | {'Holds?':<8}")
        lines.append("-" * 90)
        
        for item in data["group_1_and_2"]:
            # Distinguish between single run and bootstrap aggregate
            mode = item.get("mode", "N/A")
            type_ = item.get("type", "N/A")
            
            if "lhs_mean" in item:
                # Bootstrap
                lhs = f"{item['lhs_mean']:.4f} \u00b1 {item['lhs_std']:.4f}"
                rhs = f"{item['rhs_mean']:.4f} \u00b1 {item['rhs_std']:.4f}"
                tight = f"{item['tightness_mean']:.4f}"
                holds = "100%" if item['violation_rate'] == 0 else f"{100*(1-item['violation_rate']):.1f}%"
                lines.append(f"{mode:<15} | {type_:<20} | {lhs:<10} | {rhs:<12} | {tight:<10} | {holds:<8}")
            else:
                # Single run
                lhs = f"{item['lhs']:.4f}"
                rhs = f"{item['rhs']:.4f}"
                tight = f"{item['tightness']:.4f}"
                holds = "Yes" if item['holds'] else "No"
                lines.append(f"{mode:<15} | {type_:<20} | {lhs:<10} | {rhs:<12} | {tight:<10} | {holds:<8}")
        lines.append("")

    # Group 3
    if "group_3" in data:
        lines.append("--------------------------------------------------")
        lines.append("TEST GROUP 3: SENSITIVITY ANALYSIS")
        lines.append("--------------------------------------------------")
        lines.append(f"{'Param':<15} | {'Value':<10} | {'Deferral Rate':<15} | {'Tightness':<10} | {'Holds?':<8}")
        lines.append("-" * 75)
        
        for item in data["group_3"]:
            param = item.get("param", "N/A")
            val = str(item.get("value", "N/A"))
            defer = f"{item.get('deferral_rate', 0.0):.4f}"
            tight = f"{item.get('tightness', 0.0):.4f}"
            holds = "Yes" if item.get('holds') else "No"
            lines.append(f"{param:<15} | {val:<10} | {defer:<15} | {tight:<10} | {holds:<8}")
            
    with open(txt_path, 'w') as f:
        f.write("\n".join(lines))
        
    print(f"Text report generated at {txt_path}")

if __name__ == "__main__":
    main()
