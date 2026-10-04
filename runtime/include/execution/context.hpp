#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "execution/program.hpp"
#include "operators/operator_base.hpp"

// A build-time value handle; stores graph metadata without owning tensor memory.
struct ExecutionValueHandle {
    std::string name;
    std::vector<size_t> shape;
};

// ExecutionBuilder appends values and nodes to an ExecutionProgram.
// The subsequent planner handles memory reuse and workspace allocation.
template <typename T>
class ExecutionBuilder {
   public:
    using Value = ExecutionValueHandle;

    explicit ExecutionBuilder(size_t alignment = 256) : alignment_(alignment) {}

    Value call(const std::string& op_name, const std::string& out_name,
               std::vector<size_t> shape, std::vector<Value> inputs,
               const std::string& semantic = "") {
        auto op = require_registered_operator(op_name);
        if (shape.empty()) {
            op::OperatorShapeList inferred =
                op->infer_output_shapes(to_shapes(inputs), {});
            if (inferred.size() != 1) {
                throw std::runtime_error(
                    "ExecutionBuilder requires a single inferred output shape for " +
                    op_name);
            }
            shape = std::move(inferred.front());
        }
        if (op->behavior().output_count != 1) {
            throw std::runtime_error("ExecutionBuilder call expects one output for operator: " +
                                     op_name);
        }
        Value out = ensure_value(out_name, std::move(shape));
        append_node(op, op_name, semantic, inputs, {out});
        return out;
    }

    Value call_inplace(const std::string& op_name, const Value& inout,
                       std::vector<Value> inputs,
                       const std::string& semantic = "") {
        auto op = require_registered_operator(op_name);
        if (!op->behavior().supports_inplace) {
            throw std::runtime_error("ExecutionBuilder inplace call not supported for operator: " +
                                     op_name);
        }
        append_node(op, op_name, semantic, inputs, {inout});
        return inout;
    }

    void call_void(const std::string& op_name, std::vector<Value> inputs,
                   const std::string& semantic = "") {
        auto op = require_registered_operator(op_name);
        const auto behavior = op->behavior();
        if (behavior.output_count != 0 && !behavior.has_side_effect) {
            throw std::runtime_error("ExecutionBuilder void call is invalid for operator: " +
                                     op_name);
        }
        append_node(op, op_name, semantic, inputs, {});
    }

    const ExecutionProgram& program() const {
        return program_;
    }

   private:
    Value ensure_value(const std::string& name, std::vector<size_t> shape) {
        if (name.empty()) {
            throw std::runtime_error("ExecutionBuilder value name must not be empty");
        }
        if (shape.empty()) {
            throw std::runtime_error("ExecutionBuilder value shape must not be empty for " +
                                     name);
        }

        auto it = values_.find(name);
        if (it != values_.end()) {
            if (it->second.shape != shape) {
                throw std::runtime_error("ExecutionBuilder shape mismatch for value " +
                                         name);
            }
            return it->second;
        }

        if (!known_names_.insert(name).second) {
            throw std::runtime_error("ExecutionBuilder duplicate value: " + name);
        }

        program_.values.push_back({name, std::move(shape), alignment_});
        Value value{program_.values.back().name, program_.values.back().shape};
        values_.emplace(name, value);
        return value;
    }

    void append_node(const std::shared_ptr<op::OperatorBase>& op,
                     const std::string& op_name, const std::string& semantic,
                     const std::vector<Value>& inputs,
                     const std::vector<Value>& outputs) {
        op::OperatorShapeList input_shapes;
        op::OperatorShapeList output_shapes;
        input_shapes.reserve(inputs.size());
        output_shapes.reserve(outputs.size());
        for (const auto& input : inputs) {
            input_shapes.push_back(input.shape);
        }
        for (const auto& output : outputs) {
            output_shapes.push_back(output.shape);
        }
        const size_t workspace_bytes =
            op->infer_workspace_bytes(input_shapes, output_shapes);
        const size_t workspace_alignment = op->workspace_alignment();

        ExecutionNode node{op_name, semantic, {}, {}, workspace_bytes,
                           workspace_alignment};
        node.inputs.reserve(inputs.size());
        node.outputs.reserve(outputs.size());
        for (const auto& input : inputs) {
            assert_known_value(input.name);
            node.inputs.push_back(input.name);
        }
        for (const auto& output : outputs) {
            assert_known_value(output.name);
            node.outputs.push_back(output.name);
        }
        program_.nodes.push_back(std::move(node));
    }

    void assert_known_value(const std::string& name) const {
        if (known_names_.find(name) == known_names_.end()) {
            throw std::runtime_error("ExecutionBuilder unknown value: " + name);
        }
    }

    static std::shared_ptr<op::OperatorBase> require_registered_operator(
        const std::string& op_name) {
        auto op = op::OperatorRegistry<T>::instance().getOperatorByName(op_name);
        if (!op) {
            throw std::runtime_error("ExecutionBuilder unknown operator: " + op_name);
        }
        return op;
    }

    static op::OperatorShapeList to_shapes(const std::vector<Value>& values) {
        op::OperatorShapeList shapes;
        shapes.reserve(values.size());
        for (const auto& value : values) {
            shapes.push_back(value.shape);
        }
        return shapes;
    }

    size_t alignment_ = 256;
    ExecutionProgram program_;
    std::unordered_map<std::string, Value> values_;
    std::unordered_set<std::string> known_names_;
};
