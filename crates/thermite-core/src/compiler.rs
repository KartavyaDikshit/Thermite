use crate::tree::{DecisionTreeClassifier, DecisionTreeRegressor, TreeNode};

pub fn compile_tree_c(nodes: &[TreeNode], node_idx: usize, indent: usize, is_classifier: bool) -> String {
    if nodes.is_empty() {
        return "    return 0.0;\n".to_string();
    }
    
    let node = &nodes[node_idx];
    let indent_str = " ".repeat(indent * 4);
    
    if node.is_leaf() {
        if is_classifier {
            let mut best_class = 0;
            let mut best_prob = -1.0;
            for (i, &prob) in node.value.iter().enumerate() {
                if prob > best_prob {
                    best_prob = prob;
                    best_class = i;
                }
            }
            return format!("{}return {};\n", indent_str, best_class);
        } else {
            return format!("{}return {:?};\n", indent_str, node.value[0]);
        }
    }
    
    let mut code = String::new();
    
    if node.is_categorical {
        let cats = node.left_categories.iter().map(|c| format!("X[{}] == {:?}", node.feature_idx, c)).collect::<Vec<_>>().join(" || ");
        if cats.is_empty() {
            code.push_str(&format!("{}if (0) {{\n", indent_str));
        } else {
            code.push_str(&format!("{}if ({}) {{\n", indent_str, cats));
        }
    } else {
        let nan_check = if node.nan_go_left {
            format!("isnan(X[{}]) || ", node.feature_idx)
        } else {
            "".to_string()
        };
        code.push_str(&format!("{}if ({}X[{}] <= {:?}) {{\n", indent_str, nan_check, node.feature_idx, node.threshold));
    }
    
    code.push_str(&compile_tree_c(nodes, node.left, indent + 1, is_classifier));
    code.push_str(&format!("{}}} else {{\n", indent_str));
    code.push_str(&compile_tree_c(nodes, node.right, indent + 1, is_classifier));
    code.push_str(&format!("{}}}\n", indent_str));
    
    code
}

pub fn compile_forest_regressor_c(trees: &[DecisionTreeRegressor]) -> String {
    let mut code = String::new();
    for (i, tree) in trees.iter().enumerate() {
        code.push_str(&format!("double tree_{}(const double* X) {{\n", i));
        code.push_str(&compile_tree_c(&tree.nodes, 0, 1, false));
        code.push_str("}\n\n");
    }
    code.push_str("double predict(const double* X) {\n");
    code.push_str("    double sum = 0.0;\n");
    for i in 0..trees.len() {
        code.push_str(&format!("    sum += tree_{}(X);\n", i));
    }
    code.push_str(&format!("    return sum / {:.1};\n", trees.len() as f64));
    code.push_str("}\n");
    code
}

pub fn compile_forest_classifier_c(trees: &[DecisionTreeClassifier], n_classes: usize) -> String {
    let mut code = String::new();
    for (i, tree) in trees.iter().enumerate() {
        code.push_str(&format!("int tree_{}(const double* X) {{\n", i));
        code.push_str(&compile_tree_c(&tree.nodes, 0, 1, true));
        code.push_str("}\n\n");
    }
    code.push_str("int predict(const double* X) {\n");
    code.push_str(&format!("    int counts[{}] = {{0}};\n", n_classes));
    for i in 0..trees.len() {
        code.push_str(&format!("    counts[tree_{}(X)]++;\n", i));
    }
    code.push_str("    int best_c = 0;\n    int best_v = -1;\n");
    code.push_str(&format!("    for (int i = 0; i < {}; i++) {{\n", n_classes));
    code.push_str("        if (counts[i] > best_v) {\n            best_v = counts[i];\n            best_c = i;\n        }\n    }\n");
    code.push_str("    return best_c;\n}\n");
    code
}

pub fn compile_boosting_regressor_c(trees: &[DecisionTreeRegressor], learning_rate: f64, initial_pred: f64) -> String {
    let mut code = String::new();
    for (i, tree) in trees.iter().enumerate() {
        code.push_str(&format!("double tree_{}(const double* X) {{\n", i));
        code.push_str(&compile_tree_c(&tree.nodes, 0, 1, false));
        code.push_str("}\n\n");
    }
    code.push_str("double predict(const double* X) {\n");
    code.push_str(&format!("    double sum = {:?};\n", initial_pred));
    for i in 0..trees.len() {
        code.push_str(&format!("    sum += {:?} * tree_{}(X);\n", learning_rate, i));
    }
    code.push_str("    return sum;\n}\n");
    code
}

pub fn compile_boosting_classifier_c(trees: &[DecisionTreeRegressor], learning_rate: f64, initial_pred: f64) -> String {
    let mut code = String::new();
    for (i, tree) in trees.iter().enumerate() {
        code.push_str(&format!("double tree_{}(const double* X) {{\n", i));
        code.push_str(&compile_tree_c(&tree.nodes, 0, 1, false));
        code.push_str("}\n\n");
    }
    code.push_str("double predict_proba(const double* X) {\n");
    code.push_str(&format!("    double sum = {:?};\n", initial_pred));
    for i in 0..trees.len() {
        code.push_str(&format!("    sum += {:?} * tree_{}(X);\n", learning_rate, i));
    }
    code.push_str("    return 1.0 / (1.0 + exp(-sum));\n}\n\n");
    code.push_str("int predict(const double* X) {\n");
    code.push_str("    return predict_proba(X) >= 0.5 ? 1 : 0;\n}\n");
    code
}
