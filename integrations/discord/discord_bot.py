"""MEOKCLAW Discord Bot — Sovereign AI for Your Server

Commands:
    /ask <question>      — Route through dual-brain
    /council <question>  — Multi-model consensus
    /arena <question>    — Side-by-side comparison
    /cost <model>        — Show cost for a model
    /stats               — Server usage stats

Setup:
    1. Create bot at https://discord.com/developers/applications
    2. Copy token to .env as DISCORD_BOT_TOKEN
    3. Invite bot with scopes: bot, applications.commands
    4. Run: python discord_bot.py
"""
from __future__ import annotations

import os
import asyncio
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from dual_brain_orchestrator import DualBrainOrchestrator
from openrouter_client import OpenRouterClient

BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not BOT_TOKEN:
    print("⚠️  Set DISCORD_BOT_TOKEN in .env")
    raise SystemExit(1)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
orch: Optional[DualBrainOrchestrator] = None


@bot.event
async def on_ready():
    global orch
    orch = DualBrainOrchestrator()
    await bot.tree.sync()
    print(f"🤖 MEOKCLAW Bot logged in as {bot.user}")
    print(f"   Guilds: {len(bot.guilds)}")
    print(f"   Commands: /ask, /council, /arena, /cost, /stats")


@bot.tree.command(name="ask", description="Ask the dual-brain AI")
@app_commands.describe(question="What do you want to know?")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    
    result = await orch.think(question, None)
    
    embed = discord.Embed(
        title="🧠 Dual-Brain Response",
        description=result.text[:1900],
        color=0x00D4AA if result.hemisphere == "left" else 0x8B5CF6,
    )
    embed.add_field(name="Hemisphere", value=result.hemisphere.upper(), inline=True)
    embed.add_field(name="Model", value=result.model, inline=True)
    embed.add_field(name="Cost", value=f"${result.cost_usd:.6f}", inline=True)
    embed.add_field(name="Latency", value=f"{result.latency_ms}ms", inline=True)
    embed.add_field(name="Confidence", value=f"{result.confidence:.0%}", inline=True)
    
    await interaction.followup.send(embed=embed)


@bot.tree.command(name="council", description="Consult the council (multi-model consensus)")
@app_commands.describe(question="Question for the council")
async def council_cmd(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:3201/api/council",
            json={"prompt": question, "models": ["deepseek-v4-flash", "deepseek-v4-pro", "kimi-k2.6"]}
        ) as resp:
            data = await resp.json()
    
    embed = discord.Embed(
        title="🏛️ Council Consensus",
        description=data["consensus_text"][:1900],
        color=0x8B5CF6,
    )
    embed.add_field(name="Agreement", value=f"{data['consensus_score']:.0%}", inline=True)
    embed.add_field(name="Total Cost", value=f"${data['total_cost_usd']:.6f}", inline=True)
    embed.add_field(name="Models", value=str(len(data["models"])), inline=True)
    
    if data["disagreeing_models"]:
        embed.add_field(name="Dissent", value=", ".join(data["disagreeing_models"]), inline=False)
    
    await interaction.followup.send(embed=embed)


@bot.tree.command(name="arena", description="Run models side-by-side")
@app_commands.describe(question="Prompt for the arena")
async def arena_cmd(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:3201/api/arena",
            json={"prompt": question, "models": ["deepseek-v4-flash", "deepseek-v4-pro"]}
        ) as resp:
            data = await resp.json()
    
    embed = discord.Embed(
        title="⚔️ Model Arena",
        description=f"**Prompt:** {question[:200]}",
        color=0xF59E0B,
    )
    
    for m in data["models"]:
        winner = " 👑" if m["model"] == data.get("winner") else ""
        embed.add_field(
            name=f"{m['model']}{winner}",
            value=f"${m['cost_usd']:.6f} • {m['latency_ms']}ms\n{m['text'][:300]}...",
            inline=False,
        )
    
    await interaction.followup.send(embed=embed)


@bot.tree.command(name="cost", description="Compare model costs")
@app_commands.describe(model="Model name (deepseek-flash, deepseek-pro, kimi, gpt4)")
async def cost_cmd(interaction: discord.Interaction, model: str):
    rates = {
        "deepseek-flash": (0.0001, 0.0002),
        "deepseek-pro": (0.0015, 0.005),
        "kimi": (0.002, 0.008),
        "gpt4": (0.005, 0.015),
    }
    inp, out = rates.get(model.lower(), (0.001, 0.003))
    
    embed = discord.Embed(title=f"💰 {model} Cost Structure", color=0x10B981)
    embed.add_field(name="Input", value=f"${inp}/1K tokens", inline=True)
    embed.add_field(name="Output", value=f"${out}/1K tokens", inline=True)
    embed.add_field(name="1K In + 500 Out", value=f"${(inp + out * 0.5):.4f}", inline=True)
    
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="stats", description="Show server usage stats")
async def stats_cmd(interaction: discord.Interaction):
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:3201/health") as resp:
            health = await resp.json()
    
    embed = discord.Embed(title="📊 MEOKCLAW Status", color=0x3B82F6)
    embed.add_field(name="Version", value=health["version"], inline=True)
    embed.add_field(name="Status", value=health["status"], inline=True)
    embed.add_field(name="Features", value=", ".join(health["features"]), inline=False)
    
    await interaction.response.send_message(embed=embed)


if __name__ == "__main__":
    bot.run(BOT_TOKEN)
