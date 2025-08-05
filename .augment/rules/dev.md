---
type: "manual"
---

## 🏗️ **Generic Development Rules**

### **1. Data Integrity**
- **Always use real data** - No synthetic or fabricated datasets
- **Avoid academic misconduct** - Never fabricate results or manipulate data
- **Use legitimate sources** - Only use established benchmark datasets and real experimental data
- **Maintain data provenance** - Document data sources and preprocessing steps

### **2. Python Environment**
- **Always use local conda Python environment** - Stick to conda for package management
- **For GPU work only**: Use `module load cuda/12.1` when CUDA is required
- **Consistent environment** - Use the same Python environment across all development
- **Package management**: Use `conda install` or `pip install` within conda environment

### **3. File Management**
- **Don't generate many new test files** - Work with existing file structure
- **Revise existing files instead** - Modify current files rather than creating many new ones
- **Maintain clean directory structure** - Don't create unnecessary nested directories
- **Reuse existing frameworks** - Build on current codebase rather than starting from scratch

### **4. Pre-Submission Housekeeping**
- **Delete unnecessary files** before submitting jobs to cluster:
  - Temporary files (`.tmp`, `.cache`, `__pycache__/`)
  - Large intermediate results that can be regenerated
  - Old log files and debug outputs
  - Duplicate or backup files
- **Clean up results directories** - Remove outdated experimental results
- **Push to GitHub** - Always commit and push changes before cluster submission
- **Verify file sizes** - Ensure no large unnecessary files are being submitted

### **5. Version Control Workflow**
```bash
# Before every cluster job submission:
git add .
git commit -m "Prepare for cluster job: [brief description]"
git push origin main

# Clean up unnecessary files
rm -rf __pycache__/ *.tmp *.cache
rm old_results_*.json
```

### **6. Code Development Standards**
- **Modify existing scripts** - Update current files rather than creating new ones
- **Use existing models and frameworks** - Build on established codebase
- **Maintain backward compatibility** - Don't break existing functionality
- **Follow established patterns** - Use existing code style and structure

### **7. Academic Integrity**
- **No result fabrication** - All results must come from actual experiments
- **No data manipulation** - Use datasets as provided from legitimate sources
- **Proper attribution** - Credit all methods, datasets, and prior work
- **Transparent methodology** - Document all procedures clearly

### **8. Cluster Job Management**
- **Resource efficiency** - Request appropriate resources, not excessive
- **Clean workspace** - Remove unnecessary files before job submission
- **Version control** - Always push code to GitHub before submitting
- **Monitor job status** - Check job progress and handle failures appropriately

### **9. Development Workflow**
1. **Modify existing files** instead of creating new ones
2. **Test locally** with conda Python environment
3. **Clean up workspace** - delete unnecessary files
4. **Commit and push** to GitHub
5. **Submit job** to cluster
6. **Monitor results** and iterate

---

**Core Principles**: 
- Use real data to avoid academic misconduct
- Use local conda Python environment  
- For GPU: `module load cuda/12.1`
- Revise existing files, don't create many new ones
- Always clean up and push to GitHub before cluster submission
