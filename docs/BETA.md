# Beta test

Thank you for trying makaar. It turns photos and scans of notes into Markdown, LaTeX and Word on your own laptop. Nothing is uploaded. This test takes about 30 minutes, most of it waiting.

## 1. Install

- **Windows:** run `makaar_…_x64-setup.exe`. Windows will warn that the app is from an unknown publisher (it isn't signed yet): click **More info → Run anyway**.
- **macOS:** open the `.dmg` and drag makaar to Applications. The first time, right-click the app → **Open**, because it isn't signed yet.
- **Linux:** the `.AppImage` (make it executable) or the `.deb`.

On first start, click **Download models** (about 1.9 GB, once). If your notes are handwritten, also click **Get the handwriting adapter** (48 MB).

## 2. Convert one real document

Use something you would actually convert: lecture notes, a problem set, an exam paper, 5–20 pages. Photos from your phone are fine.

1. **Choose images or PDFs…** and pick the pages.
2. Tick **Handwritten notes** if they are handwritten.
3. **Convert**, and leave the laptop plugged in. Expect about a minute a page.
4. **Review flagged blocks**: the red boxes are where makaar is unsure. Fix what's wrong and save.
5. Open the Word or LaTeX file and look at it.

## 3. Tell us

Fill in the feedback form (link from whoever invited you), or [open a beta feedback issue](https://github.com/Satyanshgaur/makaar/issues/new?template=beta-feedback.md). The questions:

1. Laptop: brand and model, and price range if you know it.
2. Operating system.
3. What you converted: type, number of pages, printed or handwritten, photos or scans.
4. Seconds per page: shown in the progress list (or the `speed:` line if you used the command line).
5. How many blocks were flagged, and roughly how many of the unflagged ones were wrong anyway?
6. Did the Word file open correctly in Microsoft Word? Did the maths look right?
7. Anything that crashed, froze or confused you.
8. **Would you use makaar again for this instead of Gemini or ChatGPT?** Yes / No / Maybe, and why.

Don't send pages you can't share. If something failed, the log is in the output folder under `.makaar/llama-server.log`; attach it only if you're comfortable that it contains nothing private (it contains no page text).
