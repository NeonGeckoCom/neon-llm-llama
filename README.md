# NeonAI LLM FastChat
Proxies API calls to FastChat.

## Request Format
API requests should include `history`, a list of tuples of strings, and the current
`query`

>Example Request:
>```json
>{
>  "history": [["user", "hello"], ["llm", "hi"]],
>  "query": "how are you?"
>}
>```

## Response Format
Responses will be returned as dictionaries. Responses should contain the following:
- `response` - String LLM response to the query

## Docker Configuration
When running this as a docker container, the `XDG_CONFIG_HOME` envvar is set to `/config`.
A configuration file at `/config/neon/diana.yaml` is required and should look like:
```yaml
MQ:
  port: <MQ Port>
  server: <MQ Hostname or IP>
  users:
    mq-fastchat-api:
      password: <neon_fastchat user's password>
      user: neon_fastchat
FastChat:
  model: "fastchat"
  context_depth: 3
  max_tokens: 256
```

For example, if your configuration resides in `~/.config`:
```shell
export CONFIG_PATH="/home/${USER}/.config"
docker run -v ${CONFIG_PATH}:/config neon_llm_fastchat
```
> Note: If connecting to a local MQ server, you may need to specify `--network host`