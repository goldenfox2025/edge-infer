#pragma once

#include <string>
#include <vector>

#include "execution/workspace_liveness.hpp"

// 图里的逻辑值。shape/alignment 会直接进入 workspace 规划。
struct ExecutionValue {
    std::string name;
    std::vector<size_t> shape;
    size_t alignment = 256;
};

// 图里的一个节点。semantic 用来在 eager executor 中定位具体处理逻辑。
struct ExecutionNode {
    std::string op_name;
    std::string semantic;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    size_t workspace_bytes = 0;
    size_t workspace_alignment = 256;
};

// 轻量级执行图：只描述 value 与 node 的依赖关系。
struct ExecutionProgram {
    std::vector<ExecutionValue> values;
    std::vector<ExecutionNode> nodes;
};

template <typename T>
WorkspacePlan build_workspace_plan_from_execution_program(const ExecutionProgram& program) {
    // 先把执行图转成 liveness builder，再统一走 workspace planner。
    WorkspaceLivenessBuilder builder;
    for (const auto& value : program.values) {
        builder.add_value(value.name, value.shape, value.alignment);
    }
    for (const auto& node : program.nodes) {
        builder.add_step(node.inputs, node.outputs);
    }
    return builder.build_plan<T>();
}
