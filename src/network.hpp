#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>

#include <vector>
#include <array>
#include <chrono>
#include <random>
#include <unordered_map>

namespace dcnn
{
    struct Node
    {
        double bias;

        std::vector<std::pair<Node*, double>> out_conns;
        std::vector<std::pair<Node*, double>> in_conns;

        double (*function)(double x);
        double (*derivative)(double x);

        double activation;

        int8_t node_type = 0;
    };

    class Network
    {
    public:
        Network() = delete;
        ~Network() = default;

        explicit Network(double (*activation_function)(double x), double (*activation_function_derivative)(double x),
                         uint32_t input_nodes = 4, uint32_t output_nodes = 3, uint32_t hidden_nodes = 12,
                         uint32_t min_connections = 2, uint32_t max_connections = 12, double learning_rate = 0.01f)
        {
            m_activation_function = activation_function;
            m_activation_function_derivative = activation_function_derivative;

            m_input_nodes_c = input_nodes;
            m_output_nodes_c = output_nodes;
            m_hidden_nodes_c = hidden_nodes;

            m_min_connections = min_connections;
            m_max_connections = max_connections;

            m_input_nodes.resize(m_input_nodes_c);
            m_output_nodes.resize(m_output_nodes_c);
            m_hidden_nodes.resize(m_hidden_nodes_c);

            generate_random_input_nodes();
            generate_random_hidden_nodes();
            generate_random_output_nodes();
        }
    
        void propagate_forward(std::vector<double>* inputs, std::vector<double>* outputs)
        {
            m_inputs = *inputs;
            
            activate_network();

            *outputs = m_outputs;
        }
        void propagate_backward(std::vector<double>* target)
        {
            std::vector<double> sqr_error;
            for(uint32_t i = 0; i < m_outputs.size(); i++)
            {
                sqr_error.push_back(pow(target->at(i) - m_outputs[i], 2));
            }

            std::vector<std::vector<double>> temp_weight_gradient(m_output_nodes_c);
            std::vector<double> temp_bias_gradient(m_output_nodes_c);
            std::vector<std::vector<double>> temp_alpha_gradient1(m_output_nodes_c);

            std::vector<std::vector<double>> temp_alpha_gradient2;

            for(uint32_t i = 0; i < m_output_nodes_c; i++)
            {
                node_gradient(sqr_error[i], &m_output_nodes[i], &temp_weight_gradient[i],
                              &temp_bias_gradient[i], &temp_alpha_gradient1[i]);
            }

            m_weight_gradient.push_back(temp_weight_gradient);
            m_bias_gradient.push_back(temp_bias_gradient);

            /*printf("");

            for(uint32_t i = 0; i < m_output_nodes_c; i++)
            {
                temp_weight_gradient.resize(0);
                temp_bias_gradient.resize(0);

                for(uint32_t j = 0; j < m_output_nodes[i].in_conns.size(); j++)
                {
                    auto& n_in_conns = m_output_nodes[i].in_conns;

                    temp_weight_gradient.resize(n_in_conns.size());
                    temp_bias_gradient.resize(n_in_conns.size());
                    temp_alpha_gradient2.resize(n_in_conns.size());

                    node_gradient(temp_alpha_gradient1[i][j], n_in_conns[j].first,
                                  &temp_weight_gradient[j], &temp_bias_gradient[j],
                                  &temp_alpha_gradient2[j]);
                }

                m_weight_gradient.push_back(temp_weight_gradient);
                m_bias_gradient.push_back(temp_bias_gradient);
            }

            printf("");*/
        
            
        }

    private:

#pragma region random

        template<typename T>
        static inline T timed_seed()
        {
            return
            (T)
            std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() +
            (T)
            std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        }

        template<typename T>
        static inline T rand_range(T min, T max)
        {
            std::default_random_engine eng(timed_seed<uint64_t>());
            
            if(std::is_integral<T>())
            {
                std::uniform_int_distribution<T> dist(min, max);
                return dist(eng);
            }
            else if(std::is_floating_point<T>())
            {
                std::uniform_real_distribution<T> dist(min, max);
                return dist(eng);
            }
            else
            {
                throw std::runtime_error("[ERROR] rand_range<T> requires T to be integral or floating point.\n");
            }
        }

        template<typename T>
        static void uniform_rand_vector(std::vector<T>* out, T min, T max, uint32_t sz)
        {
            std::mt19937 rng(timed_seed<uint64_t>());

            if(std::is_integral<T>())
            {
                std::uniform_int_distribution<T> dist(min, max);
                for(uint32_t i = 0; i < sz; i++)
                {
                    out->push_back(dist(rng));
                }
            }
            else if(std::is_floating_point<T>())
            {
                std::uniform_real_distribution<T> dist(min, max);
                for(uint32_t i = 0; i < sz; i++)
                {
                    out->push_back(dist(rng));
                }
            }
            else
            {
                throw std::runtime_error("[ERROR] uniform_rand_vector<T> requires T to be integral or floating point.\n");
            }
        }

#pragma endregion

#pragma region util

        template<typename T>
        static inline bool find_in_vec(std::vector<T>* vec, T value)
        {
            auto found = std::find(vec->begin(), vec->end(), value);
            return found != vec->end();
        }
        
        template<typename T, typename F>
        static inline bool find_pair_in_vec(std::vector<std::pair<T, F>>* vec, T value)
        {
            for(uint32_t i = 0; i < vec->size(); i++)
            {
                if(vec->at(i).first == value) return true;
            }
            return false;
        }

#pragma endregion

#pragma region math

        template<typename T>
        static inline void mat_vec_mult(std::vector<std::vector<T>>* mat, std::vector<T>* vec,
                                        std::vector<std::vector<T>>* out, bool vertical = false)
        {
            if(vertical)
            {
                if(mat->size() != vec->size()) throw std::runtime_error("[ERROR] Cannot multiply unequal size vectors.\n");
                
                out->resize(mat->size());
                for(uint32_t i = 0; i < mat->at(0).size(); i++)
                {
                    for(uint32_t j = 0; j < mat->size(); j++)
                    {
                        out->at(j).push_back(mat->at(j)[i] * vec->at(j));
                    }
                }
            }
            else
            {
                if(mat->at(0).size() != vec->size()) throw std::runtime_error("[ERROR] Cannot multiply unequal size vectors.\n");
                out->resize(mat->size());
                for(uint32_t i = 0; i < mat->size(); i++)
                {
                    for(uint32_t j = 0; j < mat->at(0).size(); j++)
                    {
                        out->at(i).push_back(mat->at(i)[j] * vec->at(j));
                    }
                }
            }
        }

        template<typename T>
        static inline void mat_vec_add(std::vector<std::vector<T>>* mat, std::vector<T>* vec,
                                       std::vector<std::vector<T>>* out, bool vertical = false)
        {
            if(vertical)
            {
                if(mat->size() != vec->size()) throw std::runtime_error("[ERROR] Cannot add unequal size vectors.\n");
                
                out->resize(mat->size());
                for(uint32_t i = 0; i < mat->at(0).size(); i++)
                {
                    for(uint32_t j = 0; j < mat->size(); j++)
                    {
                        out->at(j).push_back(mat->at(j)[i] + vec->at(j));
                    }
                }
            }
            else
            {
                if(mat->at(0).size() != vec->size()) throw std::runtime_error("[ERROR] Cannot add unequal size vectors.\n");
                out->resize(mat->size());
                for(uint32_t i = 0; i < mat->size(); i++)
                {
                    for(uint32_t j = 0; j < mat->at(0).size(); j++)
                    {
                        out->at(i).push_back(mat->at(i)[j] + vec->at(j));
                    }
                }
            }
        }

        template<typename T>
        static inline void vec_scl_mult(std::vector<T>* vec, T val,
                                        std::vector<T>* out)
        {
            for(uint32_t i = 0; i < vec->size(); i++)
            {
                out->push_back(vec->at(i) * val);
            }
        }

        template<typename T>
        static inline void vec_scl_add(std::vector<T>* vec, T val,
                                       std::vector<T>* out)
        {
            for(uint32_t i = 0; i < vec->size(); i++)
            {
                out->push_back(vec->at(i) + val);
            }
        }

        template<typename T>
        static inline void vec_vec_mult(std::vector<T>* vec, std::vector<T>* val,
                                        std::vector<T>* out)
        {
            if(vec->size() != val->size()) throw std::runtime_error("[ERROR] Cannot multiply unequal size vectors.\n");
            for(uint32_t i = 0; i < vec->size(); i++)
            {
                out->push_back(vec->at(i) * val->at(i));
            }
        }

        template<typename T>
        static inline void vec_vec_add(std::vector<T>* vec, std::vector<T>* val,
                                       std::vector<T>* out)
        {
            if(vec->size() != val->size()) throw std::runtime_error("[ERROR] Cannot add unequal size vectors.\n");
            for(uint32_t i = 0; i < vec->size(); i++)
            {
                out->push_back(vec->at(i) + val->at(i));
            }
        }

        template<typename T>
        static inline void vec_inner_add(std::vector<T>* vec, T* out)
        {
            for(uint32_t i = 0; i < vec->size(); i++)
            {
                *out += vec->at(i);
            }
        }

        template<typename T>
        static inline void vec_inner_mult(std::vector<T>* vec, T* out)
        {
            for(uint32_t i = 0; i < vec->size(); i++)
            {
                *out *= vec->at(i);
            }
        }

        void node_gradient(double target, Node* node, std::vector<double>* weight_gradient,
                           double* bias_gradient, std::vector<double>* alpha_gradient)
        {
            auto eta = [](Node* nd) -> double
            {
                double temp_activation = 0.0f;
                for(uint32_t i = 0; i < nd->in_conns.size(); i++)
                {
                    temp_activation += nd->in_conns[i].first->activation * nd->in_conns[i].second;
                }

                return temp_activation + nd->bias;
            };

            for(uint32_t i = 0; i < node->in_conns.size(); i++)
            {
                *bias_gradient += node->derivative(eta(node->in_conns[i].first)) * 2 * (node->activation - target);
                weight_gradient->push_back(node->in_conns[i].first->activation * node->derivative(eta(node->in_conns[i].first)) * 2 * (node->activation - target));
                alpha_gradient->push_back(node->in_conns[i].second * node->derivative(eta(node->in_conns[i].first)) * 2 * (node->activation - target));
            }
        }

#pragma endregion

        void generate_random_input_nodes()
        {
            for(auto& node : m_input_nodes)
            {
                node.function = m_activation_function;
                node.derivative = m_activation_function_derivative;
                node.node_type = 1;

                uint32_t conns = rand_range(m_min_connections, m_max_connections);
                node.out_conns.reserve(conns);
                for(uint32_t i = 0; i < conns; i++)
                {
                    auto val = m_hidden_nodes_c > 1 ? rand_range<uint32_t>(0, m_hidden_nodes_c - 1) : 0;
                    auto dt = &m_hidden_nodes[val];

                    if(!find_pair_in_vec(&node.out_conns, dt))
                    {
                        std::pair<Node*, double> Dt;
                        Dt.first = dt;
                        Dt.second = rand_range<double>(-1.0f, 1.0f);
                        node.bias = rand_range<double>(-10.0f, 10.0f);

                        node.out_conns.push_back(Dt);
                        dt->in_conns.push_back(std::pair(&node, Dt.second));

                        m_connected_nodes.push_back(dt);
                    }
                }
            }
            for(auto& inode: m_input_nodes)
            {
                m_connected_nodes.push_back(&inode);
            }
        }
        void generate_random_hidden_nodes()
        {
            for(auto& node: m_hidden_nodes)
            {
                node.function = m_activation_function;
                node.derivative = m_activation_function_derivative;

                uint32_t conns = rand_range(m_min_connections, m_max_connections);
                node.in_conns.reserve(conns);
                for(uint32_t i = 0; i < conns; i++)
                {
                    auto val = rand_range<uint32_t>(0, m_connected_nodes.size() - 1);
                    auto dt = m_connected_nodes[val];

                    if(!find_pair_in_vec(&node.in_conns, dt))
                    {
                        std::pair<Node*, double> Dt;
                        Dt.first = dt;
                        Dt.second = rand_range<double>(-1.0f, 1.0f);
                        node.bias = rand_range<double>(-10.0f, 10.0f);

                        node.in_conns.push_back(Dt);
                        dt->out_conns.push_back(std::pair(&node, Dt.second));

                        m_connected_nodes.push_back(&node);
                    }
                }
            }
        }
        void generate_random_output_nodes()
        {
            for(auto& node : m_output_nodes)
            {
                node.function = m_activation_function;
                node.derivative = m_activation_function_derivative;
                node.node_type = -1;

                uint32_t conns = rand_range(m_min_connections, m_max_connections);
                node.in_conns.reserve(conns);
                for(uint32_t i = 0; i < conns; i++)
                {
                    auto val = m_hidden_nodes_c > 1 ? rand_range<uint32_t>(0, m_hidden_nodes_c - 1) : 0;
                    auto dt = &m_hidden_nodes[val];
                    if(!find_pair_in_vec(&node.in_conns, dt))
                    {
                        std::pair<Node*, double> Dt;
                        Dt.first = dt;
                        Dt.second = rand_range<double>(-1.0f, 1.0f);
                        node.bias = rand_range<double>(-10.0f, 10.0f);

                        node.in_conns.push_back(Dt);
                        dt->out_conns.push_back(std::pair(&node, Dt.second));

                        m_connected_nodes.push_back(&node);
                    }
                }
            }
        }

        void activate_node(Node* node)
        {
            static uint32_t input_node_c = 0;
            if(node->node_type == 1)
            {
                if(input_node_c >= m_input_nodes.size()) throw std::runtime_error("[ERROR] Too many input nodes activated!\n");
                node->activation = node->function(m_inputs[input_node_c]);
                input_node_c++;
            }
            else if(node->node_type == -1)
            {
                double temp_activation = 0.0f;
                for(uint32_t i = 0; i < node->in_conns.size(); i++)
                {
                    temp_activation += node->in_conns[i].first->activation * node->in_conns[i].second;
                }

                node->activation = node->function(temp_activation + node->bias);
                m_outputs.push_back(node->activation);
            }
            else
            {
                double temp_activation = 0.0f;
                for(uint32_t i = 0; i < node->in_conns.size(); i++)
                {
                    temp_activation += node->in_conns[i].first->activation * node->in_conns[i].second;
                }

                node->activation = node->function(temp_activation + node->bias);
            }
        }
        void activate_network()
        {
            for(auto& node : m_input_nodes) activate_node(&node);
            for(auto& node : m_hidden_nodes) activate_node(&node);
            for(auto& node : m_output_nodes) activate_node(&node);
        }

        double (*m_activation_function)(double x);
        double (*m_activation_function_derivative)(double x);

        uint32_t m_input_nodes_c = 4;
        uint32_t m_output_nodes_c = 3;
        uint32_t m_hidden_nodes_c = 12;

        uint32_t m_min_connections = 2;
        uint32_t m_max_connections = 12;

        double learning_rate = 0.01f;

        std::vector<Node> m_input_nodes;
        std::vector<Node> m_output_nodes;
        std::vector<Node> m_hidden_nodes;

        std::vector<Node*> m_connected_nodes;

        std::vector<double> m_inputs;
        std::vector<double> m_outputs;

        std::vector<std::vector<std::vector<double>>> m_weight_gradient;
        std::vector<std::vector<double>> m_bias_gradient;
    };
}