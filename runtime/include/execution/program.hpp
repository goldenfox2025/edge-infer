#pragma once

#include <string>
#include <vector>

#include "execution/workspace_liveness.hpp"

// A logical graph value; shape and alignment feed workspace planning.
struct ExecutionValue {
    std::string name;
    std::vector<size_t> shape;
    size_t alignment = 256;
};

// A graph node; semantic identifies its implementation in the eager executor.
struct ExecutionNode {
    std::string op_name;
    std::string semantic;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    size_t workspace_bytes = 0;
    size_t workspace_alignment = 256;
};

// An execution graph describing dependencies between values and nodes.
struct ExecutionProgram {
    std::vector<ExecutionValue> values;
    std::vector<ExecutionNode> nodes;
};

template <typename T>
WorkspacePlan build_workspace_plan_from_execution_program(const ExecutionProgram& program) {
    // Translate the graph into lifetime intervals, then use the shared workspace planner.
    WorkspaceLivenessBuilder builder;
    for (const auto& value : program.values) {
        builder.add_value(value.name, value.shape, value.alignment);
    }
    for (const auto& node : program.nodes) {
        builder.add_step(node.inputs, node.outputs);
    }
    return builder.build_plan<T>();
}
