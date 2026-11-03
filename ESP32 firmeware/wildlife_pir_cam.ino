#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_http_server.h"

// ============================================================
// ESP32-CAM AI Thinker: SoftAP + MJPEG stream + periodic JPEG uploads
// Stream: http://192.168.4.1/stream
// Laptop: 192.168.4.2, Flask: http://192.168.4.2:5000
// ============================================================

const char* AP_SSID = "Wildlife_Monitor";
const char* AP_PASSWORD = "Wildlife@123";
const char* LAPTOP_IP = "192.168.4.2";
constexpr uint16_t SERVER_PORT = 5000;

IPAddress apIP(192, 168, 4, 1);
IPAddress gateway(192, 168, 4, 1);
IPAddress subnet(255, 255, 255, 0);

const uint32_t FRAME_UPLOAD_INTERVAL_MS = 2000;  // ESP32 sends one JPEG every 2 seconds
const uint32_t UPLOAD_TIMEOUT_MS = 5000;

// AI Thinker ESP32-CAM pin map
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

httpd_handle_t camera_httpd = nullptr;
uint32_t lastFrameUpload = 0;
uint32_t uploadedFrames = 0;

bool initCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;  // 320x240 for stable Wi-Fi streaming
  config.jpeg_quality = 14;           // lower number = higher image quality
  config.fb_count = psramFound() ? 2 : 1;
  if (psramFound()) {
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_LATEST;
  } else {
    config.frame_size = FRAMESIZE_QQVGA;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }
  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor) sensor->set_framesize(sensor, psramFound() ? FRAMESIZE_QVGA : FRAMESIZE_QQVGA);
  Serial.println("Camera initialized.");
  Serial.printf("PSRAM: %s\n", psramFound() ? "yes" : "no");
  return true;
}

static esp_err_t streamHandler(httpd_req_t* req) {
  static const char* BOUNDARY = "wildlife-frame-boundary";
  char partHeader[96];
  esp_err_t res = httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=wildlife-frame-boundary");
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store, no-cache, must-revalidate");
  httpd_resp_set_hdr(req, "Connection", "close");

  Serial.println("Dashboard connected to /stream");
  while (true) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Stream frame capture failed");
      return ESP_FAIL;
    }
    if (fb->format != PIXFORMAT_JPEG) {
      esp_camera_fb_return(fb);
      continue;
    }

    res = httpd_resp_send_chunk(req, "--wildlife-frame-boundary\r\n", strlen("--wildlife-frame-boundary\r\n"));
    if (res == ESP_OK) {
      int hlen = snprintf(partHeader, sizeof(partHeader),
                          "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
                          (unsigned int)fb->len);
      res = httpd_resp_send_chunk(req, partHeader, hlen);
    }
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, "\r\n", 2);
    esp_camera_fb_return(fb);
    if (res != ESP_OK) {
      Serial.println("Stream client disconnected.");
      break;
    }
    vTaskDelay(pdMS_TO_TICKS(80)); // ~8-12 FPS depending on camera/network
  }
  return res;
}

bool startCameraStreamServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.ctrl_port = 32768;
  config.max_open_sockets = 4;
  config.stack_size = 8192;
  config.lru_purge_enable = true;

  if (httpd_start(&camera_httpd, &config) != ESP_OK) {
    Serial.println("Could not start camera HTTP server.");
    return false;
  }
  httpd_uri_t streamUri = {};
  streamUri.uri = "/stream";
  streamUri.method = HTTP_GET;
  streamUri.handler = streamHandler;
  streamUri.user_ctx = nullptr;
  if (httpd_register_uri_handler(camera_httpd, &streamUri) != ESP_OK) {
    Serial.println("Could not register /stream route.");
    httpd_stop(camera_httpd);
    camera_httpd = nullptr;
    return false;
  }
  Serial.println("MJPEG server started: http://192.168.4.1/stream");
  return true;
}

bool startAccessPoint() {
  WiFi.mode(WIFI_AP);
  WiFi.setSleep(false);
  if (!WiFi.softAPConfig(apIP, gateway, subnet)) return false;
  if (!WiFi.softAP(AP_SSID, AP_PASSWORD, 1, false, 1)) return false;
  Serial.printf("AP ready: %s\nESP32 IP: %s\n", AP_SSID, WiFi.softAPIP().toString().c_str());
  return true;
}

bool uploadFrame(camera_fb_t* fb) {
  if (!fb || fb->format != PIXFORMAT_JPEG) return false;
  if (WiFi.softAPgetStationNum() == 0) {
    Serial.println("Laptop not connected; skipping periodic frame upload.");
    return false;
  }
  String url = String("http://") + LAPTOP_IP + ":" + SERVER_PORT + "/api/frame";
  WiFiClient client;
  HTTPClient http;
  http.setConnectTimeout(4000);
  http.setTimeout(UPLOAD_TIMEOUT_MS);
  if (!http.begin(client, url)) return false;
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-Filename", "esp32_frame.jpg");
  http.addHeader("X-Device", "esp32-cam-periodic");
  int status = http.POST(fb->buf, fb->len);
  Serial.printf("Frame upload HTTP status: %d (%u bytes)\n", status, (unsigned int)fb->len);
  if (status > 0) Serial.println(http.getString());
  else Serial.printf("Upload error: %s\n", http.errorToString(status).c_str());
  http.end();
  return status >= 200 && status < 300;
}

void captureAndUpload() {
  if (WiFi.softAPgetStationNum() == 0) {
    Serial.println("Laptop not connected; skipping frame upload.");
    return;
  }

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Periodic frame capture failed.");
    return;
  }

  bool ok = uploadFrame(fb);
  esp_camera_fb_return(fb);
  if (ok) {
    uploadedFrames++;
    Serial.printf("Frame %lu sent to backend.\n", (unsigned long)uploadedFrames);
  } else {
    Serial.println("Frame upload failed; will retry on next interval.");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1200);
  Serial.println("\nWildlife monitor: MJPEG video + 2-second frame push + backend YOLO");
  if (!initCamera()) while (true) delay(1000);
  if (!startAccessPoint()) while (true) delay(1000);
  if (!startCameraStreamServer()) while (true) delay(1000);
  lastFrameUpload = millis() - FRAME_UPLOAD_INTERVAL_MS; // send first frame immediately
  Serial.println("Connect laptop to Wildlife_Monitor; laptop Wi-Fi IPv4: 192.168.4.2");
  Serial.println("Live stream: http://192.168.4.1/stream");
  Serial.println("Uploading one JPEG every 2 seconds to /api/frame.");
  Serial.println("YOLO inference frequency is controlled by the dashboard/backend.");
}

void loop() {
  const uint32_t now = millis();
  if (now - lastFrameUpload >= FRAME_UPLOAD_INTERVAL_MS) {
    lastFrameUpload = now;
    captureAndUpload();
  }
  delay(5);
}
