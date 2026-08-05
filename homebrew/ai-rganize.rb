# Homebrew formula for AI-rganize.
#
# Recommended: host this file in a tap repo named `homebrew-airganize`:
#   https://github.com/adefemi171/homebrew-airganize
#   Formula/ai-rganize.rb  (copy of this file)
#
# Users then install with:
#   brew tap adefemi171/airganize
#   brew install ai-rganize
#
# Or from this repo (development):
#   brew install --formula ./homebrew/ai-rganize.rb
#
# After cutting a release tag, refresh sha256:
#   ./scripts/update_homebrew_sha.sh v1.0.0
class AiRganize < Formula
  desc "AI-powered file organizer for your local folders"
  homepage "https://github.com/adefemi171/ai-rganize"
  # Prefer PyPI once the first release is published; GitHub tag is the bootstrap source.
  url "https://github.com/adefemi171/ai-rganize/archive/refs/tags/v1.0.0.tar.gz"
  sha256 "343d7615b0bf27aa9aad72332c7b8b5bc9560e7e68d93376aaa2ee33536d9089"
  license "MIT"
  head "https://github.com/adefemi171/ai-rganize.git", branch: "main"

  depends_on "python@3.12"
  depends_on "ffmpeg" => :recommended

  def install
    python = Formula["python@3.12"].opt_libexec/"bin/python"
    python = Formula["python@3.12"].opt_bin/"python3.12" unless python.exist?

    system python, "-m", "venv", libexec
    system libexec/"bin/pip", "install", "--upgrade", "pip"
    # Install this source tree (GitHub archive / HEAD checkout) with GUI extras.
    system libexec/"bin/pip", "install", ".[gui]"

    bin.install_symlink libexec/"bin/ai-rganize"
    bin.install_symlink libexec/"bin/ai-organize"
    bin.install_symlink libexec/"bin/ai-rganize-gui"
    bin.install_symlink libexec/"bin/ai-rganize-permissions"
  end

  def caveats
    <<~EOS
      Set a provider API key before organizing, for example:
        export OPENAI_API_KEY="..."

      Preview changes safely:
        ai-rganize organize -d ~/Downloads --dry-run --no-council
    EOS
  end

  test do
    assert_match "AI-rganize", shell_output("#{bin}/ai-rganize --help")
  end
end
