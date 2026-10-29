# Chat screen: reviewer checklist

- The composer's **+** menu holds uploads, Tools, Plan first, Manage tools, Use documents (with scope) and
  Memories; active ones show as chips beside it. Tools and chat are one place, like other chat apps.
- **Stop** must halt a reply at any moment, including while a long prompt is still being read (the engine
  checks between chunks). Look for states where the stop button is hidden or does nothing.
- Every reply shows tokens/s and where it ran (graphics card, split, processor, drive) afterwards.
- The model picker lists every chat model on disk, grouped by family, labelled `name · quant · publisher`;
  models that cannot run yet are greyed with the reason, never hidden.
- A model's saved settings (presets) keep Thinking and a custom effort; switching models and back must
  restore them exactly (`lib/chat/presets.ts`).
- DeepSeek-V4.1 uses its own chat format (`<｜User｜>…<｜Assistant｜>` with `<think>`/`</think>`): check that
  re-rendering a conversation produces the same tokens the model generated, or every turn re-reads the
  whole history (the engine only continues an exact prefix for this model).
- Long conversations: no quadratic re-render of the message list while tokens stream.
