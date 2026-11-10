import { tool } from "@opencode-ai/plugin"

export default tool({
  description: "Save this workspace's single candidate: two instruction strings.",
  args: {
    name_instructions: tool.schema.string(),
    parameter_instructions: tool.schema.string(),
  },
  async execute(args, context) {
    const child = Bun.spawn([
      process.env.CMM_PYTHON!, "-m", "tools.prompt_tools", "write_prompt", context.directory,
    ], {
      cwd: process.env.CMM_REPO,
      env: process.env,
      stdin: "pipe", stdout: "pipe", stderr: "pipe", signal: context.abort,
    })
    child.stdin.write(JSON.stringify(args))
    child.stdin.end()
    const [output, error, code] = await Promise.all([
      new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited,
    ])
    if (code !== 0) throw new Error(error)
    return output
  },
})
