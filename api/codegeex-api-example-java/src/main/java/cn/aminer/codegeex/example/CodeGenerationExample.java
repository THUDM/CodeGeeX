package cn.aminer.codegeex.example;

import cn.aminer.codegeex.example.pojo.Payload;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;

import java.io.IOException;

/**
 * 调用 CodeGeeX API 生成代码的例子。
 *
 * @author Darran Zhang @ codelast.com
 * @version 2023-01-20
 */
public class CodeGenerationExample {
  public static final String API_KEY = "your_api_key";  // 在"天启开放平台"上申请到的API Key
  public static final String API_SECRET = "your_api_secret";  // 在"天启开放平台"上申请到的API Secret
  public static final int NUMBER = 3;  // 生成几个候选
  public static final String LANGUAGE = "Java";  // 编程语言
  public static final String REQUEST_URL = "https://tianqi.aminer.cn/api/v2/multilingual_code_generate";  // 请求地址

  public static void main(String[] args) throws Exception {
    CodeGenerationExample example = new CodeGenerationExample();
    String prompt = "// use OkHttpClient library to write a function to perform http post request\n\n" +
      "public class HttpPost {\n" +
      "    public static void main(String[] args) {\n";
    example.generateCode(prompt);
  }

  /**
   * 生成代码。
   *
   * @param prompt 待补全的代码
   */
  public void generateCode(String prompt) throws Exception {
    ObjectMapper objectMapper = new ObjectMapper();
    Payload payload = new Payload().setApiKey(API_KEY).setApiSecret(API_SECRET).setPrompt(prompt).setNumber(NUMBER)
      .setLanguage(LANGUAGE);
    String response = performHttpPost(REQUEST_URL, objectMapper.writeValueAsString(payload));
    System.out.println(response);
  }

  /**
   * 发起 HTTP POST 请求。
   *
   * @param url     请求的URL
   * @param payload 请求的JSON数据
   * @return 请求返回的内容，若出错则返回 null。
   */
  public String performHttpPost(String url, String payload) {
    HttpUrl.Builder builder = null;
    try {
      HttpUrl httpUrl = HttpUrl.parse(url);
      if (httpUrl != null) {
        builder = httpUrl.newBuilder();
      }
    } catch (IllegalArgumentException e) {
      System.out.println("failed to create HttpUrl.Builder from url " + url + ":" + e);
    }
    if (builder == null) {
      return null;
    }
    OkHttpClient client = new OkHttpClient();
    RequestBody requestBody = RequestBody.create(payload, MediaType.parse("application/json; charset=utf-8"));
    Request request = new Request.Builder()
      .url(builder.build())
      .post(requestBody)
      .build();

    try {
      Response response = client.newCall(request).execute();
      ResponseBody body = response.body();
      if (body == null) {
        System.out.println("null response body");
        return null;
      }
      return body.string();
    } catch (IOException e) {
      System.out.println("failed to send POST request: " + e);
    }
    return null;
  }
}
