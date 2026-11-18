const messages = document.querySelector('#messages');
const form = document.querySelector('#composer');
const input = document.querySelector('#question');
const send = document.querySelector('#send');
let activeRefundBubble = null;

// 创建一条用户或客服消息，返回气泡元素供追加商品卡片。
function message(text, role, pending = false) {
  const row = document.createElement('div');
  row.className = `message ${role}`;
  const bubble = document.createElement('div');
  bubble.className = `bubble${pending ? ' pending' : ''}`;
  bubble.textContent = text;
  row.appendChild(bubble);
  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

// 使用 DOM API 在指定客服气泡内创建商品卡片，避免推荐内容出现在对话框外。
function showProducts(items, bubble) {
  if (!Array.isArray(items) || items.length === 0) return;
  const list = document.createElement('div');
  list.className = 'products';
  list.setAttribute('aria-label', '本轮推荐商品');
  for (const item of items) {
    const card = document.createElement('article');
    card.className = 'product-card';
    const visual = document.createElement('div');
    visual.className = 'product-image';
    // 仅允许以 https:// 开头的图片地址，避免把其他协议当作商品图片加载。
    if (typeof item.image === 'string' && /^https:\/\//i.test(item.image)) {
      const image = document.createElement('img');
      image.src = item.image;
      image.alt = item.name || '商品图片';
      image.loading = 'lazy';
      // 图片加载失败时移除图片元素，并显示文字占位。
      image.onerror = () => { image.remove(); visual.textContent = '暂无图片'; };
      visual.appendChild(image);
    } else {
      const placeholder = document.createElement('span');
      placeholder.className = 'image-placeholder';
      placeholder.textContent = '暂无图片';
      visual.appendChild(placeholder);
    }
    const detail = document.createElement('div');
    detail.className = 'product-details';
    const title = document.createElement('h3');
    title.textContent = item.name || '商品';
    const tags = document.createElement('div');
    tags.className = 'tags';
    for (const label of [item.qualityDesc, ...(item.labels || [])].filter(Boolean).slice(0, 4)) {
      const tag = document.createElement('span');
      tag.className = 'tag';
      tag.textContent = label;
      tags.appendChild(tag);
    }
    const price = document.createElement('div');
    price.className = 'price';
    if (typeof item.dailyRental === 'number') {
      const strong = document.createElement('strong');
      strong.textContent = `¥${item.dailyRental.toFixed(2)}`;
      price.append(strong, ' / 天起');
    } else {
      price.textContent = '租金以商品页面为准';
    }
    detail.append(title, tags, price);
    card.append(visual, detail);
    list.appendChild(card);
  }
  bubble.classList.add('with-products');
  bubble.appendChild(list);
}

function showOrders(orders, bubble) {
  if (!Array.isArray(orders) || !orders.length) return;
  const section = document.createElement('section');
  section.className = 'order-list';
  const heading = document.createElement('strong');
  heading.textContent = '模拟订单（仅供演示）';
  section.appendChild(heading);
  for (const order of orders) {
    const line = document.createElement('div');
    line.textContent = `${order.id} · ${order.name} · ${order.memory} · ¥${Number(order.dailyRental).toFixed(2)}/天 · ${order.status}`;
    section.appendChild(line);
  }
  bubble.appendChild(section);
}

function renderResult(result, bubble) {
  bubble.textContent = result.answer || '当前没有对话';
  bubble.classList.remove('pending');
  showProducts(result.products, bubble);
  showOrders(result.orders, bubble);
  if (!result.pending_refund || !result.quote) {
    activeRefundBubble = null;
    return;
  }
  activeRefundBubble = bubble;
  const actions = document.createElement('div');
  actions.className = 'refund-actions';
  const detail = document.createElement('p');
  const quote = result.quote;
  detail.textContent = quote.mode === 'exchange'
    ? `模拟换单：订单 ${quote.order_id} → ${quote.product_name}，日租金 ¥${Number(quote.old_daily_rental).toFixed(2)} → ¥${Number(quote.new_daily_rental).toFixed(2)}。`
    : `模拟退款：订单 ${quote.order_id}，日租金 ¥${Number(quote.old_daily_rental).toFixed(2)}；不创建新订单。`;
  actions.appendChild(detail);
  for (const [label, approved] of [[quote.mode === 'exchange' ? '确认模拟退款并换单' : '确认模拟退款', true], ['暂不退款', false]]) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.addEventListener('click', () => decideRefund(approved));
    actions.appendChild(button);
  }
  bubble.appendChild(actions);
}

async function decideRefund(approved) {
  if (!activeRefundBubble) return;
  const bubble = activeRefundBubble;
  const buttons = bubble.querySelectorAll('.refund-actions button');
  buttons.forEach(button => { button.disabled = true; });
  try {
    const response = await fetch('api/refund/confirm', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approved }),
    });
    if (!response.ok) throw new Error('确认失败，请刷新页面查看会话状态。');
    const result = await response.json();
    renderResult(result, bubble);
  } catch (error) {
    buttons.forEach(button => { button.disabled = false; });
    const message = document.createElement('p');
    message.textContent = error.message;
    bubble.appendChild(message);
  }
}

// 提交用户问题、调用聊天 API，并把回答和商品卡片追加到当前对话。
async function ask(question) {
  const text = question.trim();
  if (!text || send.disabled) return;
  input.value = '';
  send.disabled = true;
  message(text, 'user');
  const bubble = message('正在为你查找合适的商品…', 'assistant', true);
  try {
    const response = await fetch('api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    if (!response.ok) throw new Error('商品服务暂时不可用，请稍后重试。');
    const result = await response.json();
    renderResult(result, bubble);
  } catch (error) {
    bubble.textContent = error.message || '请求失败，请稍后重试。';
    bubble.classList.remove('pending');
  } finally {
    send.disabled = false;
    input.focus();
    messages.scrollTop = messages.scrollHeight;
  }
}

// 表单提交时阻止页面刷新，改为发送当前输入的问题。
form.addEventListener('submit', event => { event.preventDefault(); ask(input.value); });
// 为每个快捷问题按钮绑定点击事件，点击后直接发送对应问题。
document.querySelectorAll('[data-question]').forEach(button => {
  button.addEventListener('click', () => ask(button.dataset.question));
});

// Refresh can restore a pending confirmation from the SQLite checkpoint.
fetch('api/session').then(response => response.json()).then(result => {
  if (result.pending_refund) renderResult(result, message('', 'assistant'));
}).catch(() => {});
