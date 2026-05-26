import * as vscode from 'vscode';

interface MeokclawConfig {
    apiBase: string;
    defaultMode: string;
}

function getConfig(): MeokclawConfig {
    const cfg = vscode.workspace.getConfiguration('meokclaw');
    return {
        apiBase: cfg.get<string>('apiBase', 'http://localhost:3201'),
        defaultMode: cfg.get<string>('defaultMode', 'auto'),
    };
}

async function askMeokclaw(prompt: string, mode?: string): Promise<void> {
    const config = getConfig();
    const panel = vscode.window.createWebviewPanel(
        'meokclawResponse',
        'MEOKCLAW Response',
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );

    panel.webview.html = getLoadingHtml(prompt);

    try {
        const modelId = mode ? `meokclaw-${mode}` : `meokclaw-${config.defaultMode}`;
        const res = await fetch(`${config.apiBase}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: modelId,
                messages: [{ role: 'user', content: prompt }],
            }),
        });

        const data = await res.json();
        panel.webview.html = getResponseHtml(data);
    } catch (err) {
        panel.webview.html = getErrorHtml(String(err));
    }
}

function getLoadingHtml(prompt: string): string {
    return `<!DOCTYPE html>
<html>
<head><style>
    body { font-family: system-ui; padding: 20px; color: #ccc; background: #1e1e1e; }
    .loading { display: flex; align-items: center; gap: 10px; }
    .spinner { width: 20px; height: 20px; border: 2px solid #444; border-top-color: #00D4AA; border-radius: 50%; animation: spin 1s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
</style></head>
<body>
    <div class="loading"><div class="spinner"></div>Routing via Corpus Callosum...</div>
    <p style="opacity:0.6; font-size:12px; margin-top:10px;">${escapeHtml(prompt)}</p>
</body></html>`;
}

function getResponseHtml(data: any): string {
    const text = data.choices?.[0]?.message?.content || '[No response]';
    const cost = data.usage?.cost_usd?.toFixed(6) || '?';
    const savings = data.usage?.savings_vs_gpt4?.toFixed(1) || '?';
    const meta = data.meokclaw_meta || {};
    const hemisphere = meta.hemisphere || 'unknown';
    const latency = meta.latency_ms || '?';

    return `<!DOCTYPE html>
<html>
<head><style>
    body { font-family: system-ui; padding: 20px; color: #ccc; background: #1e1e1e; line-height: 1.6; }
    .meta { display: flex; gap: 15px; margin-bottom: 15px; font-size: 11px; }
    .badge { padding: 2px 8px; border-radius: 4px; background: #333; }
    .savings { color: #00D4AA; font-weight: bold; }
    .content { white-space: pre-wrap; font-size: 13px; }
    hr { border: none; border-top: 1px solid #333; margin: 15px 0; }
</style></head>
<body>
    <div class="meta">
        <span class="badge">Hemisphere: ${hemisphere.toUpperCase()}</span>
        <span class="badge">Cost: $${cost}</span>
        <span class="badge savings">Savings: ${savings}% vs GPT-4</span>
        <span class="badge">Latency: ${latency}ms</span>
    </div>
    <hr>
    <div class="content">${escapeHtml(text)}</div>
</body></html>`;
}

function getErrorHtml(err: string): string {
    return `<!DOCTYPE html><html><body style="font-family:system-ui;padding:20px;color:#ff6b6b;background:#1e1e1e;">
        <h3>Error</h3><p>${escapeHtml(err)}</p>
        <p style="opacity:0.6;font-size:12px;">Make sure MEOKCLAW API is running on localhost:3201</p>
    </body></html>`;
}

function escapeHtml(text: string): string {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export function activate(context: vscode.ExtensionContext) {
    context.subscriptions.push(
        vscode.commands.registerCommand('meokclaw.ask', async () => {
            const prompt = await vscode.window.showInputBox({
                prompt: 'Ask MEOKCLAW anything...',
                placeHolder: 'e.g., Explain this error...',
            });
            if (prompt) await askMeokclaw(prompt);
        }),
        vscode.commands.registerCommand('meokclaw.explain', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;
            const selection = editor.document.getText(editor.selection);
            if (!selection) {
                vscode.window.showWarningMessage('Select some code first');
                return;
            }
            await askMeokclaw(`Explain this code:\n\n${selection}`, 'right');
        }),
        vscode.commands.registerCommand('meokclaw.council', async () => {
            const prompt = await vscode.window.showInputBox({
                prompt: 'Ask the council (multi-model consensus)...',
            });
            if (prompt) await askMeokclaw(prompt, 'council');
        }),
    );
}

export function deactivate() {}
