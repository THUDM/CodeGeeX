package cn.aminer.codegeex.example.pojo;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.experimental.Accessors;

/**
 * 发送到 CodeGeex API 的请求中包含的JSON payload对象。
 *
 * @author Darran Zhang @ codelast.com
 * @version 2023-01-20
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@Data
@Accessors(chain = true)
public class Payload {
  @JsonProperty("apikey")
  String apiKey;  // 在"天启开放平台"上申请到的API Key

  @JsonProperty("apisecret")
  String apiSecret;  // 在"天启开放平台"上申请到的API Secret

  String prompt;  // 待补全的代码

  @JsonProperty("n")
  int number;  // 生成几个候选

  @JsonProperty("lang")
  String language;  // 编程语言
}
