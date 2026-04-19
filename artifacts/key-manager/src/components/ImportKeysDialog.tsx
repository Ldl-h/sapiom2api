import { useState } from "react";
import { useImportKeys, getListKeysQueryKey, getGetKeyStatsQueryKey } from "@workspace/api-client-react";
import { useQueryClient } from "@tanstack/react-query";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { Label } from "@/components/ui/label";

export function ImportKeysDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const [input, setInput] = useState("");
  const importKeys = useImportKeys();
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const handleImport = () => {
    if (!input.trim()) return;

    let keysData: Array<{ key: string; name?: string; provider?: string; note?: string }> = [];

    try {
      // Try parsing as JSON first
      const parsed = JSON.parse(input);
      if (Array.isArray(parsed)) {
        keysData = parsed.map(item => ({
          key: item.key,
          name: item.name || "Imported Key",
          provider: item.provider,
          note: item.note
        })).filter(k => k.key);
      } else {
        throw new Error("JSON must be an array");
      }
    } catch (e) {
      // Fallback: split by newlines AND commas, then clean up
      const tokens = input
        .split(/[\n,]+/)
        .map(s => s.trim())
        .filter(Boolean);
      keysData = tokens.map(token => ({
        key: token,
        name: "Imported Key"
      }));
    }

    if (keysData.length === 0) {
      toast({
        title: "Invalid Format",
        description: "Could not parse any keys from the input.",
        variant: "destructive",
      });
      return;
    }

    importKeys.mutate(
      { data: { keys: keysData } },
      {
        onSuccess: (res) => {
          queryClient.invalidateQueries({ queryKey: getListKeysQueryKey() });
          queryClient.invalidateQueries({ queryKey: getGetKeyStatsQueryKey() });
          toast({
            title: "Import Successful",
            description: `Imported ${res.imported} keys. Skipped ${res.skipped}.`,
          });
          setInput("");
          onOpenChange(false);
        },
        onError: () => {
          toast({
            title: "Import Failed",
            description: "An error occurred while importing keys.",
            variant: "destructive",
          });
        },
      }
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Import Keys</DialogTitle>
          <DialogDescription>
            Paste your keys here. Supports comma-separated, one-per-line, or JSON array format.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-2">
          <div className="space-y-2">
            <Label>Keys Input</Label>
            <Textarea
              className="font-mono text-sm min-h-[200px] resize-none"
              placeholder={"sk_live_123...\nsk_live_456...\n\n// Or JSON:\n[\n  { \"key\": \"sk_...\", \"name\": \"Stripe\" }\n]"}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              data-testid="input-import-textarea"
            />
          </div>
        </div>
        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={() => onOpenChange(false)} data-testid="button-cancel-import">
            Cancel
          </Button>
          <Button onClick={handleImport} disabled={!input.trim() || importKeys.isPending} data-testid="button-submit-import">
            {importKeys.isPending ? "Importing..." : "Import"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
