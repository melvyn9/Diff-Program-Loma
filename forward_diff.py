import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff

def forward_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_fwd : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply forward differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', forward_diff() should return
        def d_square(x : In[_dfloat]) -> _dfloat:
            return make__dfloat(x.val * x.val, x.val * x.dval + x.dval * x.val)
        where the class _dfloat is
        class _dfloat:
            val : float
            dval : float
        and the function make__dfloat is
        def make__dfloat(val : In[float], dval : In[float]) -> _dfloat:
            ret : _dfloat
            ret.val = val
            ret.dval = dval
            return ret

        Parameters:
        diff_func_id - the ID of the returned function
        structs - a dictionary that maps the ID of a Struct to 
                the corresponding Struct
        funcs - a dictionary that maps the ID of a function to 
                the corresponding func
        diff_structs - a dictionary that maps the ID of the primal
                Struct to the corresponding differential Struct
                e.g., diff_structs['float'] returns _dfloat
        func - the function to be differentiated
        func_to_fwd - mapping from primal function ID to its forward differentiation
    """

    # HW1 happens here. Modify the following IR mutators to perform
    # forward differentiation.

    # Apply the differentiation.
    class FwdDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            # HW1: TODO
            # From lecture
            new_args = [loma_ir.Arg(
                arg.id,
                autodiff.type_to_diff_type(diff_structs, arg.t),
                arg.i) for arg in node.args]

            new_body = []
            for stmt in node.body:
                mutated = self.mutate_stmt(stmt)
                if isinstance(mutated, list):
                    new_body.extend(mutated)
                else:
                    new_body.append(mutated)

            new_node = loma_ir.FunctionDef(
                diff_func_id,
                new_args,
                new_body,
                node.is_simd,
                autodiff.type_to_diff_type(diff_structs, node.ret_type))
            return new_node

        def mutate_return(self, node):
            # HW1: TODO
            # First, mutate the return expression into a two-tuple (primal, derivative)
            val, dval = self.mutate_expr(node.val)
            # Check the type of the return expression using node.val.t.
            if isinstance(node.val.t, loma_ir.Float):
                # For floats, pack into a _dfloat.
                new_expr = loma_ir.Call('make__dfloat', [val, dval], lineno=node.lineno)
                return loma_ir.Return(new_expr, lineno=node.lineno)
            else:
                # For int (or other non-differentiable types), return only the primal value.
                return loma_ir.Return(val, lineno=node.lineno)
            # return super().mutate_return(node)

        def mutate_declare(self, node):
            # HW1: TODO
            # Convert the declared type to its differential
            diff_type = autodiff.type_to_diff_type(diff_structs, node.t)
            # Create the declaration without initialization
            declare_stmt = loma_ir.Declare(node.target, diff_type, lineno=node.lineno)

            # If there is no initializer then just return the declaration
            if node.val is None:
                return declare_stmt

            # Mutate the initializer expression
            rhs_val, rhs_dval = self.mutate_expr(node.val)

            # Check if this is a float type
            is_float = isinstance(node.t, loma_ir.Float)

            if is_float:
                # Assign to target.val and target.dval
                assign_val = loma_ir.Assign(
                    loma_ir.StructAccess(loma_ir.Var(node.target), 'val', lineno=node.lineno),
                    rhs_val,
                    lineno=node.lineno
                )
                assign_dval = loma_ir.Assign(
                    loma_ir.StructAccess(loma_ir.Var(node.target), 'dval', lineno=node.lineno),
                    rhs_dval,
                    lineno=node.lineno
                )
                return [declare_stmt, assign_val, assign_dval]
            else:
                # For non-float types only assign the primal value
                assign = loma_ir.Assign(
                    loma_ir.Var(node.target),
                    rhs_val,
                    lineno=node.lineno
                )
                return [declare_stmt, assign]

        def mutate_assign(self, node):
            # HW1: TODO
            rhs_val, rhs_dval = self.mutate_expr(node.val)
            lhs_val, lhs_dval = self.mutate_expr(node.target)

            # Check if the RHS is a float
            is_float = isinstance(node.val.t, loma_ir.Float)

            if is_float:
                # Assign both val and dval for dual number
                assign_val = loma_ir.Assign(lhs_val, rhs_val)
                assign_dval = loma_ir.Assign(lhs_dval, rhs_dval)
                return [assign_val, assign_dval]
            # Not a float so just assign the primal
            else:
                return loma_ir.Assign(lhs_val, rhs_val)

        def mutate_ifelse(self, node):
            # HW3: TODO
            return super().mutate_ifelse(node)

        def mutate_while(self, node):
            # HW3: TODO
            return super().mutate_while(node)

        def mutate_const_float(self, node):
            # HW1: TODO
            # From lecture
            return (node, loma_ir.ConstFloat(0.0))
            # return super().mutate_const_float(node)

        def mutate_const_int(self, node):
            # HW1: TODO
            # From lecture
            return (node, loma_ir.ConstFloat(0.0))
            # return super().mutate_const_int(node)

        def mutate_var(self, node):
            # HW1: TODO
            # If the variable is of a differentiated struct type then dont split.
            # Check using the "_d"
            if isinstance(node.t, loma_ir.Struct) and str(node.t).startswith("_d"):
                # Return the whole variable as the primal and no separate derivative.
                return (node, None)
            elif isinstance(node.t, loma_ir.Float):
                # For differentiable scalars return the .val and .dval
                return (
                    loma_ir.StructAccess(node, 'val', lineno=node.lineno, t=node.t),
                    loma_ir.StructAccess(node, 'dval', lineno=node.lineno, t=node.t)
                )
            else:
                # For other non-differentiable types, return the variable
                # and set the derivative to 0
                return (node, loma_ir.ConstFloat(0.0))
            # return super().mutate_var(node)

        def mutate_array_access(self, node):
            # HW1: TODO
            # Mutate the array expression to get its primal value.
            prim_array, _ = self.mutate_expr(node.array)
            # Mutate the index expression (we ignore its derivative).
            index, _ = self.mutate_expr(node.index)
            
            # Build a new ArrayAccess node using the mutated array and index
            access = loma_ir.ArrayAccess(prim_array, index, lineno=node.lineno, t=node.t)
            
            # If the element type is differentiable, then expect each element of the array
            # to be a dual number
            if isinstance(node.t, loma_ir.Float):
                # Extract the primal and derivative value from the dual
                val = loma_ir.StructAccess(access, 'val', lineno=node.lineno, t=node.t)
                dval = loma_ir.StructAccess(access, 'dval', lineno=node.lineno, t=node.t)
                return (val, dval)
            else:
                # For non-differentiable element types, just return the ArrayAccess as the value
                # and provide a zero derivative.
                return (access, loma_ir.ConstFloat(0.0))
            # return super().mutate_array_access(node)

        def mutate_struct_access(self, node):
            # HW1: TODO
            # Mutate the struct expression
            struct, _ = self.mutate_expr(node.struct)
            
            # If the type of the field is differentiable
            if isinstance(node.t, loma_ir.Float):
                # First access the field from the struct; this yields the dual (e.g., _dfloat) stored there.
                field_dual = loma_ir.StructAccess(struct, node.member_id, lineno=node.lineno, t=node.t)
                # Extract the primal and derivative value from the dual
                val = loma_ir.StructAccess(field_dual, 'val', lineno=node.lineno, t=node.t)
                dval = loma_ir.StructAccess(field_dual, 'dval', lineno=node.lineno, t=node.t)
                return (val, dval)
            else:
                # For non-differentiable fields, just access the field
                # and assign a constant derivative of 0.
                val = loma_ir.StructAccess(struct, node.member_id, lineno=node.lineno, t=node.t)
                return (val, loma_ir.ConstFloat(0.0))
            
        def mutate_add(self, node):
            # HW1: TODO
            # From lecture
            # Unpack the left and right subexpressions
            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)

            # Compute the primal and tangent parts
            new_val = loma_ir.BinaryOp(loma_ir.Add(), left_val, right_val,
                                    lineno=node.lineno, t=node.t)
            new_dval = loma_ir.BinaryOp(loma_ir.Add(), left_dval, right_dval,
                                        lineno=node.lineno, t=node.t)

            # Return a tuple
            return (new_val, new_dval)
            # return super().mutate_add(node)

        def mutate_sub(self, node):
            # HW1: TODO
            # Basically the same as add but with Sub()
            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)

            new_val = loma_ir.BinaryOp(loma_ir.Sub(), left_val, right_val,
                                    lineno=node.lineno, t=node.t)
            new_dval = loma_ir.BinaryOp(loma_ir.Sub(), left_dval, right_dval,
                                        lineno=node.lineno, t=node.t)

            return (new_val, new_dval)
            # return super().mutate_sub(node)

        def mutate_mul(self, node):
            # HW1: TODO
            # Basically the same as add but with Mul()
            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)

            new_val = loma_ir.BinaryOp(loma_ir.Mul(), left_val, right_val,
                                    lineno=node.lineno, t=node.t)
            # Product rule: left_val * right_dval + right_val * left_dval
            term1 = loma_ir.BinaryOp(loma_ir.Mul(), left_val, right_dval,
                                    lineno=node.lineno, t=node.t)
            term2 = loma_ir.BinaryOp(loma_ir.Mul(), right_val, left_dval,
                                    lineno=node.lineno, t=node.t)
            new_dval = loma_ir.BinaryOp(loma_ir.Add(), term1, term2,
                                        lineno=node.lineno, t=node.t)

            return (new_val, new_dval)
            # return super().mutate_mul(node)

        def mutate_div(self, node):
            # HW1: TODO
            # Basically the same as add but with Div()
            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)

            new_val = loma_ir.BinaryOp(loma_ir.Div(), left_val, right_val,
                                    lineno=node.lineno, t=node.t)
            # Quotient rule: (right_val * left_dval - left_val * right_dval) / (right_val * right_val)
            numerator_left = loma_ir.BinaryOp(loma_ir.Mul(), right_val, left_dval,
                                            lineno=node.lineno, t=node.t)
            numerator_right = loma_ir.BinaryOp(loma_ir.Mul(), left_val, right_dval,
                                            lineno=node.lineno, t=node.t)
            numerator = loma_ir.BinaryOp(loma_ir.Sub(), numerator_left, numerator_right,
                                        lineno=node.lineno, t=node.t)
            denominator = loma_ir.BinaryOp(loma_ir.Mul(), right_val, right_val,
                                        lineno=node.lineno, t=node.t)
            new_dval = loma_ir.BinaryOp(loma_ir.Div(), numerator, denominator,
                                        lineno=node.lineno, t=node.t)
            return (new_val, new_dval)
            # return super().mutate_div(node)

        def mutate_call(self, node):
            # HW1: TODO
            # Grab the arguments and follow the formulas provided on HW1 instructions
            new_args = [self.mutate_expr(arg) for arg in node.args]
            if node.id == 'sin':
                val, dval = new_args[0]
                primal = loma_ir.Call('sin', [val], lineno=node.lineno, t=node.t)
                tangent = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    loma_ir.Call('cos', [val], lineno=node.lineno, t=node.t),
                    dval,
                    lineno=node.lineno,
                    t=node.t
                )
                return (primal, tangent)
                    
            elif node.id == 'cos':
                # cos(x): derivative = -sin(x) * x.dval
                val, dval = new_args[0]
                primal = loma_ir.Call('cos', [val], lineno=node.lineno, t=node.t)
                neg_sin = loma_ir.BinaryOp(
                    loma_ir.Sub(),
                    loma_ir.ConstFloat(0.0),
                    loma_ir.Call('sin', [val], lineno=node.lineno, t=node.t),
                    lineno=node.lineno, t=node.t
                )
                tangent = loma_ir.BinaryOp(
                    loma_ir.Mul(), neg_sin, dval, lineno=node.lineno, t=node.t
                )
                return (primal, tangent)

            elif node.id == 'sqrt':
                # sqrt(x): derivative = x.dval / (2 * sqrt(x))
                val, dval = new_args[0]
                primal = loma_ir.Call('sqrt', [val], lineno=node.lineno, t=node.t)
                two = loma_ir.ConstFloat(2.0)
                two_sqrt = loma_ir.BinaryOp(
                    loma_ir.Mul(), two, primal, lineno=node.lineno, t=node.t
                )
                tangent = loma_ir.BinaryOp(
                    loma_ir.Div(), dval, two_sqrt, lineno=node.lineno, t=node.t
                )
                return (primal, tangent)

            elif node.id == 'pow':
                # pow(x, y): derivative =
                #   dx_term: x.dval * (y * x^(y-1))
                #   dy_term: y.dval * (x^y * log(x))
                # Total: dx_term + dy_term.
                base_val, base_dval = new_args[0]
                exp_val, exp_dval = new_args[1]
                primal = loma_ir.Call('pow', [base_val, exp_val],
                                    lineno=node.lineno, t=node.t)
                one = loma_ir.ConstFloat(1.0)
                exp_minus_1 = loma_ir.BinaryOp(
                    loma_ir.Sub(), exp_val, one, lineno=node.lineno, t=node.t
                )
                base_pow_exp_minus1 = loma_ir.Call('pow', [base_val, exp_minus_1],
                                                lineno=node.lineno, t=node.t)
                dx_term = loma_ir.BinaryOp(
                    loma_ir.Mul(), base_dval,
                    loma_ir.BinaryOp(
                        loma_ir.Mul(), exp_val, base_pow_exp_minus1,
                        lineno=node.lineno, t=node.t
                    ),
                    lineno=node.lineno, t=node.t
                )
                log_base = loma_ir.Call('log', [base_val], lineno=node.lineno, t=node.t)
                dy_term = loma_ir.BinaryOp(
                    loma_ir.Mul(), exp_dval,
                    loma_ir.BinaryOp(
                        loma_ir.Mul(), primal, log_base,
                        lineno=node.lineno, t=node.t
                    ),
                    lineno=node.lineno, t=node.t
                )
                tangent = loma_ir.BinaryOp(
                    loma_ir.Add(), dx_term, dy_term, lineno=node.lineno, t=node.t
                )
                return (primal, tangent)

            elif node.id == 'exp':
                # exp(x): derivative = exp(x) * x.dval
                val, dval = new_args[0]
                primal = loma_ir.Call('exp', [val], lineno=node.lineno, t=node.t)
                tangent = loma_ir.BinaryOp(
                    loma_ir.Mul(), dval, primal, lineno=node.lineno, t=node.t
                )
                return (primal, tangent)

            elif node.id == 'log':
                # log(x): derivative = x.dval / x
                val, dval = new_args[0]
                primal = loma_ir.Call('log', [val], lineno=node.lineno, t=node.t)
                tangent = loma_ir.BinaryOp(
                    loma_ir.Div(), dval, val, lineno=node.lineno, t=node.t
                )
                return (primal, tangent)

            elif node.id == 'int2float':
                # int2float(x): derivative is 0.
                val, _ = new_args[0]
                primal = loma_ir.Call('int2float', [val], lineno=node.lineno, t=node.t)
                tangent = loma_ir.ConstFloat(0.0)
                return (primal, tangent)

            elif node.id == 'float2int':
                # float2int(x): derivative is 0.
                val, dval = new_args[0]
                primal = loma_ir.Call('float2int', [val], lineno=node.lineno, t=node.t)
                tangent = loma_ir.ConstFloat(0.0)
                return (primal, tangent)
                # return super().mutate_call(node)

    return FwdDiffMutator().mutate_function_def(func)
