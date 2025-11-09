# Super-Powered LM Arena - Autostart Guide

## 🚀 Automatic System Loading

The Super-Powered LM Arena system is now configured to automatically start whenever you open a new terminal session!

---

## 🔧 What's Automatically Configured

### 1. **Environment Variables**
```bash
LM_ARENA_HOME="/home/gary/lm-arena"
CLAUDE_ORCHESTRATOR_HOME="/home/gary/claude-orchestrator"
MANUS_AUTOMATION_HOME="/home/gary/manus-automation-agent"
PYTHONPATH="$LM_ARENA_HOME:$PYTHONPATH"
```

### 2. **Auto-Start Script**
- **Location**: `~/start-super-powered-arena.sh`
- **Executable**: ✅ Made executable
- **Function**: Automatically starts LM Arena API on port 8999

### 3. **Shell Integration**
- **~/.bash_login**: Runs on interactive login
- **~/.profile**: Auto-starts LM Arena in interactive shells
- **~/.bashrc**: Loaded with handy aliases and functions

---

## 🎯 Available Commands (After Auto-Load)

### **Basic Commands**
```bash
arena-start          # Start LM Arena system
arena-status         # Check system health status
arena-dashboard      # Open monitoring dashboard
arena-stop           # Stop LM Arena system
arena-restart        # Restart the LM Arena system
arena-logs           # View real-time logs
arena-help           # Show command reference
```

### **Web Automation Testing**
```bash
arena-test-search    # Test search automation
arena-test-scrape    # Test website scraping
```

### **Agent Management**
```bash
orchestrator-start   # Start Claude Orchestrator
orchestrator-status  # Check orchestrator status
manus-start          # Start Manus automation agent
manus-status         # Check Manus agent status
```

---

## 🌐 Access Points

### **Web Interface**
- **📊 Dashboard**: http://localhost:8999/dashboard
- **🔍 Health Check**: http://localhost:8999/health
- **📈 Metrics**: http://localhost:8999/monitoring/metrics
- **📚 API Docs**: http://localhost:8999/docs

### **Direct API Testing**
```bash
# Test web automation
curl -X POST http://localhost:8999/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "search for weather in London", "model": "web-automation", "conversation_id": "test-123"}'

# Health check
curl http://localhost:8999/health
```

---

## 🔄 How Autostart Works

### **On New Terminal Session**
1. **Environment variables** are automatically loaded
2. **LM Arena API** starts automatically on port 8999 (if not running)
3. **Aliases** become available for easy management
4. **Status message** shows system availability

### **On System Reboot**
- Run `~/start-super-powered-arena.sh` manually or open a new terminal

---

## 📊 System Status Monitoring

### **Quick Status Check**
```bash
# Check if everything is running
arena-status

# Check all components
ps aux | grep -E "(uvicorn|orchestrator|manus)"
```

### **Log Monitoring**
```bash
# View API logs
arena-logs

# View specific logs
tail -f /tmp/lm-arena-api.log
```

---

## 🌐 Web Automation Examples

Once the system is auto-started, you can use these web automation prompts:

```bash
# Via API
curl -X POST http://localhost:8999/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "search for weather in London", "model": "web-automation", "conversation_id": "test"}'

# Or use the quick test aliases
arena-test-search    # Tests weather search
arena-test-scrape    # Tests website scraping
```

**Supported automation commands:**
- `"search for [query]"` - Google search automation
- `"scrape [URL]"` - Website content extraction
- `"weather in [city]"` - Weather information lookup
- `"extract data from website"` - General data extraction

---

## 🔧 Customization

### **Modify Autostart Behavior**
Edit `~/.profile` to change auto-start settings:

```bash
# Disable auto-start
# Comment out or remove the auto-start section

# Change startup delay
# Modify the sleep duration in the startup script

# Add custom commands
# Add your own aliases to ~/.bashrc
```

### **Add Custom Aliases**
```bash
# Edit ~/.bashrc and add:
alias my-custom-command="your-command-here"
```

---

## 🚨 Troubleshooting

### **If Auto-Start Fails**
```bash
# Check manual startup
~/start-super-powered-arena.sh

# Check logs
tail -f /tmp/lm-arena-api.log

# Verify dependencies
cd /home/gary/lm-arena && source venv/bin/activate
```

### **If Commands Not Found**
```bash
# Reload shell configuration
source ~/.bashrc

# Or open new terminal
```

### **If Port Conflicts**
```bash
# Check what's using port 8999
netstat -tlnp | grep :8999

# Kill conflicting processes
pkill -f "port 8999"
```

---

## 🎉 Success Indicators

You'll know autostart is working when you see:

```
🚀 Auto-starting Super-Powered LM Arena...
🤖 Super-Powered LM Arena Environment Loaded
==============================================
📊 Dashboard: http://localhost:8999/dashboard
🔍 Health: curl http://localhost:8999/health
💡 Type 'arena-help' for command reference
```

---

## 📝 Quick Reference

| Command | Function |
|---------|----------|
| `arena-help` | Show all available commands |
| `arena-dashboard` | Open monitoring dashboard |
| `arena-status` | Check system health |
| `arena-test-search` | Test web automation search |
| `arena-test-scrape` | Test website scraping |

---

**🎯 Your Super-Powered LM Arena is now automatically available in every terminal session!**