const form = document.getElementById("analyze-form");
const newChatButton = document.getElementById("new-chat-btn");
const userIdInput = document.getElementById("user_id");
const chatIdInput = document.getElementById("chat_id");
const aiopsResult = document.getElementById("aiops-result");
const aiopsSessionState = document.getElementById("aiops-session-state");
const aiopsTraceLogPath = document.getElementById("aiops-trace-log-path");
const llmAnswer = document.getElementById("llm-answer");
const llmTraceLogPath = document.getElementById("llm-trace-log-path");
let currentUserId = "u1001";
let currentChatId = "";
let hasConversation = false;

function renderResult(data) {  // 统一展示后端原始响应，便于联调排障时快速对照字段。
  aiopsResult.textContent = JSON.stringify(data, null, 2);
}

function renderSessionState() {  // 显示当前会话绑定的 chatId，方便确认是否续聊同一会话。
  if (!currentChatId) {
    aiopsSessionState.textContent = "chatId 暂无";
    return;
  }
  aiopsSessionState.textContent = `chatId=${currentChatId}`;
}

function renderTraceLogPath(chatId) {  // 日志文件固定按 chatId 分桶展示，便于定位同一会话全过程。
  const traceLogPath = chatId ? `.agent/logs/traces/${chatId}.log` : "未找到 chatId 日志文件";
  aiopsTraceLogPath.textContent = traceLogPath;
  llmTraceLogPath.textContent = traceLogPath;
}

function renderAIOpsLLMAnswer(data) {  // 大模型回答区只展示任务链路的总结结论，不走独立问答。
  const status = String(data.status || "").toLowerCase();
  const rootCause = String(data.rootCause || "").trim();
  const solution = String(data.solution || "").trim();
  const message = String(data.message || "").trim();
  if (status === "finished") {
    const parts = [];
    if (rootCause) {
      parts.push(`根因：${rootCause}`);
    }
    if (solution) {
      parts.push(`建议：${solution}`);
    }
    llmAnswer.textContent = parts.length ? parts.join("\n") : "任务已完成，但暂无可展示的根因与建议";
    return;
  }
  if (message) {
    llmAnswer.textContent = message;
    return;
  }
  llmAnswer.textContent = "任务进行中，请继续补充信息或稍后查询结果";
}

async function fetchChatStatus() {  // 当任务完成后再查一次状态接口，拿到根因与方案的最终结构化字段。
  if (!currentChatId) {
    return null;
  }
  const response = await fetch(`/aiops/chat/${encodeURIComponent(currentChatId)}`);
  const data = await response.json();
  renderResult(data);
  renderAIOpsLLMAnswer(data);
  return data;
}

async function createChat(userId, message) {  // 首轮请求创建会话（或复用 chatId）并返回大模型结果。
  const payload = {
    userId,
    message,
    chatId: currentChatId || "",
  };
  const response = await fetch("/aiops/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  currentChatId = data.chatId || currentChatId;
  if (chatIdInput) {
    chatIdInput.value = currentChatId;
  }
  hasConversation = true;
  renderSessionState();
  renderTraceLogPath(currentChatId);
  renderResult(data);
  renderAIOpsLLMAnswer(data);
  if (String(data.status || "").toLowerCase() === "finished") {
    await fetchChatStatus();
  }
}

async function appendMessage(userId, message) {  // 同一会话后续轮次只传 chatId + message 继续推进分析。
  const payload = {
    userId,
    chatId: currentChatId,
    message,
  };
  const response = await fetch("/aiops/message", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  currentChatId = data.chatId || currentChatId;
  if (chatIdInput) {
    chatIdInput.value = currentChatId;
  }
  hasConversation = true;
  renderSessionState();
  renderTraceLogPath(currentChatId);
  renderResult(data);
  renderAIOpsLLMAnswer(data);
  if (String(data.status || "").toLowerCase() === "finished") {
    await fetchChatStatus();
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = (document.getElementById("query").value || "").trim();
  if (!message) {
    llmAnswer.textContent = "请输入问题描述后再发送";
    return;
  }
  currentUserId = (userIdInput.value || "").trim() || "u1001";
  const inputChatId = (chatIdInput?.value || "").trim();
  if (inputChatId) {
    currentChatId = inputChatId;
  }
  if (!hasConversation) {
    await createChat(currentUserId, message);
    return;
  }
  await appendMessage(currentUserId, message);
});

newChatButton.addEventListener("click", () => {  // 主动开启新对话时清空本地 chat 绑定，让下一次提交走创建逻辑。
  hasConversation = false;
  currentChatId = "";
  if (chatIdInput) {
    chatIdInput.value = "";
  }
  renderSessionState();
  aiopsTraceLogPath.textContent = "提交分析后显示";
  llmTraceLogPath.textContent = "提交分析后显示";
  aiopsResult.textContent = "提交分析后显示";
  llmAnswer.textContent = "提交分析后显示";
});

renderSessionState();
