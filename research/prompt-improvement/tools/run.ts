import { tool } from "@opencode-ai/plugin"

export default tool({
  description: "Evaluate the saved candidate on the fixed 11 development cases.",
  args: {},
  async execute(args, context) {
    const child = Bun.spawn([
      process.env.CMM_PYTHON!, "-m", "tools.prompt_tools", "run", context.directory,
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
