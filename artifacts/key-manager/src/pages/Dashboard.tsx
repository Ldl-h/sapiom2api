import { useState } from "react";
import { StatsCards } from "@/components/StatsCards";
import { KeysTable } from "@/components/KeysTable";
import { AddKeyDialog } from "@/components/AddKeyDialog";
import { ImportKeysDialog } from "@/components/ImportKeysDialog";
import { Button } from "@/components/ui/button";
import { Plus, Download, ShieldCheck, Loader2, Trash2 } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { getListKeysQueryKey, getGetKeyStatsQueryKey } from "@workspace/api-client-react";
import { useToast } from "@/hooks/use-toast";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

const BASE_URL = import.meta.env.BASE_URL?.replace(/\/$/, "") + "/api/";

export function Dashboard() {
  const [isAddOpen, setIsAddOpen] = useState(false);
  const [isImportOpen, setIsImportOpen] = useState(false);
  const [isValidateAllOpen, setIsValidateAllOpen] = useState(false);
  const [autoBan, setAutoBan] = useState(true);
  const [onlyActive, setOnlyActive] = useState(true);
  const [isValidating, setIsValidating] = useState(false);
  const [isPurging, setIsPurging] = useState(false);
  const [isPurgeConfirmOpen, setIsPurgeConfirmOpen] = useState(false);
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const handlePurgeInvalid = async () => {
    setIsPurgeConfirmOpen(false);
    setIsPurging(true);
    try {
      const resp = await fetch(`${BASE_URL}keys/purge-invalid`, { method: "DELETE" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const result = await resp.json();
      queryClient.invalidateQueries({ queryKey: getListKeysQueryKey() });
      queryClient.invalidateQueries({ queryKey: getGetKeyStatsQueryKey() });
      toast({
        title: `已清除 ${result.deleted} 个无效 key`,
        description: '所有检测状态为「无效」的 key 已从数据库中删除。',
      });
    } catch (e) {
      toast({ title: "清除失败", description: String(e), variant: "destructive" });
    } finally {
      setIsPurging(false);
    }
  };

  const handleValidateAll = async () => {
    setIsValidateAllOpen(false);
    setIsValidating(true);
    try {
      const resp = await fetch(`${BASE_URL}keys/validate-all`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ autoBan, onlyActive, concurrency: 5 }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const result = await resp.json();
      queryClient.invalidateQueries({ queryKey: getListKeysQueryKey() });
      queryClient.invalidateQueries({ queryKey: getGetKeyStatsQueryKey() });
      toast({
        title: `检测完成，共 ${result.checked} 个 key`,
        description: [
          `有效: ${result.valid}`,
          `余额不足: ${result.noBalance}`,
          `无效: ${result.invalid}`,
          autoBan && result.banned > 0 ? `自动禁用: ${result.banned}` : null,
        ].filter(Boolean).join(" · "),
      });
    } catch (e) {
      toast({ title: "检测失败", description: String(e), variant: "destructive" });
    } finally {
      setIsValidating(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground p-4 sm:p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">API Keys</h1>
            <p className="text-muted-foreground mt-1 text-sm">管理和监控你的 Sapiom API key</p>
          </div>
          <div className="flex items-center gap-3 w-full sm:w-auto flex-wrap">
            <Button
              variant="outline"
              className="flex-1 sm:flex-none text-destructive hover:text-destructive hover:bg-destructive/10 border-destructive/30"
              onClick={() => setIsPurgeConfirmOpen(true)}
              disabled={isPurging}
              data-testid="button-purge-invalid"
            >
              {isPurging ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Trash2 className="mr-2 h-4 w-4" />
              )}
              {isPurging ? "清除中..." : "清除无效"}
            </Button>
            <Button
              variant="outline"
              className="flex-1 sm:flex-none"
              onClick={() => setIsValidateAllOpen(true)}
              disabled={isValidating}
              data-testid="button-validate-all"
            >
              {isValidating ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <ShieldCheck className="mr-2 h-4 w-4" />
              )}
              {isValidating ? "检测中..." : "全部检测"}
            </Button>
            <Button variant="outline" className="flex-1 sm:flex-none" onClick={() => setIsImportOpen(true)} data-testid="button-import-keys">
              <Download className="mr-2 h-4 w-4" />
              导入
            </Button>
            <Button className="flex-1 sm:flex-none" onClick={() => setIsAddOpen(true)} data-testid="button-add-key">
              <Plus className="mr-2 h-4 w-4" />
              添加 Key
            </Button>
          </div>
        </div>

        <StatsCards />

        <div className="bg-card rounded-xl border border-border shadow-sm overflow-hidden">
          <KeysTable />
        </div>

        <AddKeyDialog open={isAddOpen} onOpenChange={setIsAddOpen} />
        <ImportKeysDialog open={isImportOpen} onOpenChange={setIsImportOpen} />

        <AlertDialog open={isPurgeConfirmOpen} onOpenChange={setIsPurgeConfirmOpen}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>清除所有无效 Key？</AlertDialogTitle>
              <AlertDialogDescription>
                这将永久删除所有已检测为<strong>无效</strong>的 key（检测状态 = invalid）。
                <br />
                此操作不可撤销，请确保已完成"全部检测"后再执行。
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>取消</AlertDialogCancel>
              <AlertDialogAction
                onClick={handlePurgeInvalid}
                className="bg-destructive hover:bg-destructive/90 text-destructive-foreground"
              >
                确认清除
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        <AlertDialog open={isValidateAllOpen} onOpenChange={setIsValidateAllOpen}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>批量检测 Key 有效性</AlertDialogTitle>
              <AlertDialogDescription>
                将逐个调用 Sapiom API 检测每个 key 是否有效，结果会持久保存到数据库。
                <br />
                key 数量较多时检测时间较长，请耐心等待。
              </AlertDialogDescription>
            </AlertDialogHeader>
            <div className="space-y-4 py-2 px-1">
              <div className="flex items-center justify-between gap-4">
                <Label htmlFor="auto-ban" className="flex flex-col gap-1 cursor-pointer">
                  <span className="font-medium">自动禁用无效 key</span>
                  <span className="text-xs text-muted-foreground font-normal">检测为无效的 key 将自动设为禁用状态</span>
                </Label>
                <Switch id="auto-ban" checked={autoBan} onCheckedChange={setAutoBan} />
              </div>
              <div className="flex items-center justify-between gap-4">
                <Label htmlFor="only-active" className="flex flex-col gap-1 cursor-pointer">
                  <span className="font-medium">仅检测启用中的 key</span>
                  <span className="text-xs text-muted-foreground font-normal">跳过已禁用的 key，加快检测速度</span>
                </Label>
                <Switch id="only-active" checked={onlyActive} onCheckedChange={setOnlyActive} />
              </div>
            </div>
            <AlertDialogFooter>
              <AlertDialogCancel>取消</AlertDialogCancel>
              <AlertDialogAction onClick={handleValidateAll}>开始检测</AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </div>
  );
}
