import json
from pii_redactor import redact_email


test_email = """
Hello, my name is Wang Xiao-Ming.
My phone number is 0912345678.
Please send the quotation to service.demo@gmail.com.
This message is related to Project Falcon.
The delivery address is Taipei City, Xinyi District.
"""


result = redact_email(test_email)

print(json.dumps(result, ensure_ascii=False, indent=2))
