import os
from dotenv import load_dotenv
from mistralai import Mistral
import base64
from io import BytesIO


load_dotenv()


def _get_client(api_key=os.getenv("MISTRAL_API_KEY")):
    return Mistral(api_key=api_key)

def _encode_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue() 
    return base64.b64encode(img_bytes).decode("utf-8")  # Encode to base64

def get_ocr_results(img, model="pixtral-12b-2409", client=_get_client()):
    img = _encode_image_base64(img)
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                                    Task: 
                                    Generate a comprehensive markdown report from a lipid test image, accurately extracting specific lipid test parameters.
                                    
                                    
                                    ### Instructions:
                                    - **Focus on extracting the following parameters** from the image:
                                        - Triglycerides (g/L and mmol/L)
                                        - Total Cholesterol (g/L and mmol/L)
                                        - HDL Cholesterol (g/L and mmol/L)
                                        - LDL Cholesterol (g/L and mmol/L)
                                                                        
                                
                                    ### Steps to Follow:
                                    1. **Parameter Extraction**: 
                                        - Extract the value for each parameter in both g/L (gram / liter) and mmol/L (mili mol / liter), where available.
                                        - If either of the value is illegible or not present, mark it as “N/A”.
                                    
                                    2. **Unit Conversions**: 
                                        - For values measured in mg/dL (miligram / deciliter), convert to g/L by dividing by 100.
                                        - Do not process values in other units such as ui/L, micro (µ) mol/L or unrelated parameters.
                                                                    
                                    
                                    ### Output Format:
                                    Organize the output with each parameter listed on a separate line as follows:
                                    **{parameter} - {value} g/L - {value2} mmol/L**
                                    Where:
                                    - `{parameter}` is the name of the test (e.g., Total Cholesterol, HDL Cholesterol).
                                    - `{value}` is the extracted result in g/L.
                                    - `{value2}` is the converted or extracted result in mmol/L.

                                    
                                    ### Example 1:
                                        Input : An image of a lipid test report.
                                        Output:
                                            ```
                                            ### Lipid Test Report
                                            
                                            | Lipid Component     | Value (g/L) | Value (mmol/L) |
                                            |---------------------|-------------|----------------|
                                            | **Total Cholesterol** | 1.30 g/L    | 3.36 mmol/L    |
                                            | **HDL Cholesterol**   | 0.48 g/L    | 1.24 mmol/L    |
                                            | **LDL Cholesterol**   | 0.52 g/L    | 1.34 mmol/L    |
                                            | **Triglycerides**     | 1.69 g/L    | 1.93 mmol/L    |
                                            
                                            ```

                                    ### Example 2:
                                        Input : An image of a lipid test report.
                                        Output:
                                            ```
                                            ### Lipid Test Report

                                            | Lipid Component       | Value (g/L) | Value (mmol/L) |
                                            |-----------------------|-------------|----------------|
                                            | **Total Cholesterol** | 1.04 g/L    | N/A            |
                                            | **HDL Cholesterol**   | 0.65 g/L    | 1.68 mmol/L    |
                                            | **LDL Cholesterol**   | N/A         | 1.34 mmol/L    |
                                            | **Triglycerides**     | N/A         | N/A            |

                                            ```           
                                """
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{img}"
                    },
                ]
            },
        ],
        temperature=0,
    )
    
    return chat_response.choices[0].message.content[15:-3]  


def get_analysis(img, model="mistral-large-latest", client=_get_client()):
    chat_response = client.chat.complete(
    model = model,
    messages = [
        {
            "role": "user",
            "content": f"""
                        You are a top cardiologist specializing in coronary artery disease, acute coronary syndrome, and hypercholesterolemia. Your patient has provided their latest lipid test results, and you need to help them correct any parameter that is not at target in a clear and empathetic way.

                        Here are the test results:

                        {get_ocr_results(img)}

                        Output format:   don't use markdown structure and use a simple text strucutre  
                         

                        #Example:
                        ```    
                        Lipid Test Report Analysis
                        Triglycerides
                            - Value: 0.90 g/L (1.03 mmol/L)
                            - Target: Less than 1.7 mmol/L (150 mg/dL)
                            - Analysis: Your triglyceride level is within the normal range. This is good news as it indicates a lower risk of heart disease. ✅

                        Total Cholesterol
                            - Value: 1.18 g/L (3.05 mmol/L)
                            - Target: Less than 5.2 mmol/L (200 mg/dL)
                            - Analysis: Your total cholesterol level is within the desirable range. It indicates a high risk of heart disease. 🚨
                        ```
                        """,
        },
    ],
    temperature = 0
    )
    return chat_response.choices[0].message.content




def get_tips(img, model="mistral-large-latest", client=_get_client()):
    chat_response = client.chat.complete(
    model = model,
    messages = [
        {
            "role": "user",
            "content": f"""
                        You are a top cardiologist specializing in coronary artery disease, acute coronary syndrome, and hypercholesterolemia. Your patient has provided their latest lipid test results, and you need to help them correct any parameter that is not at target in a clear and empathetic way.

                        Here are the test results:

                        {get_ocr_results(img)}

                        Output format: don't use markdown structure and use a simple text strucutre  
                        #Example:
                        ```    

                        Tips for the Patient
                        Dietary Advice
                        Reduce Saturated Fats: Limit your intake of foods high in saturated fats, such as red meat, full-fat dairy products, and processed foods. 
                        Exercise and Lifestyle
                        Regular Physical Activity: Aim for at least 150 minutes of moderate-intensity aerobic exercise per week, such as brisk walking, cycling, or swimming. This can help improve your cholesterol levels and overall heart health.
                        Strength Training: Incorporate strength training exercises at least two days a week to build muscle and improve your metabolism.
                        Habits and Lifestyle Changes
                        Stay Hydrated: Drink plenty of water throughout the day to stay hydrated and support overall health.
                        Manage Stress: Practice stress-reducing activities like meditation, yoga, or deep breathing exercises to manage stress levels, which can impact heart health.
                        Things to Avoid
                        Avoid Trans Fats: Steer clear of foods containing trans fats, often found in fried foods, baked goods, and margarine.
                        Limit Processed Foods: Reduce your intake of processed and packaged foods, which are often high in sodium, sugar, and unhealthy fats.

                        By following these tips, you can work towards improving your lipid profile and overall heart health. If you have any questions or need further guidance, please don't hesitate to reach out to your healthcare provider.

                        ```

                        """,
        },
    ],
    temperature = 0
    )
    return chat_response.choices[0].message.content