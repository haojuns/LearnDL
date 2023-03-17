def generate_add_function(function_name):
    code_template = """
def {function_name}(a, b):
    return a + b
"""
    code = code_template.format(function_name=function_name)
    return code

# 生成名为 my_add 的函数代码
generated_code = generate_add_function("my_add")
print(generated_code)

# 动态执行生成的代码以定义 my_add 函数
exec(generated_code)

# 使用新生成的 my_add 函数
result = my_add(3, 5)
print(result)  # 输出 8
