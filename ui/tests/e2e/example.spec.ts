import { test, expect } from '@playwright/test'

test('download CSV and switch country', async ({ page }) => {
  await page.goto('http://localhost:3000')
  await page.waitForTimeout(500)
  // Country selector present
  const select = page.locator('select#country')
  await expect(select).toBeVisible()
  // Switch to KE
  await select.selectOption('KE')
  await page.waitForTimeout(500)
  // Download CSV
  const [ download ] = await Promise.all([
    page.waitForEvent('download'),
    page.locator('button:has-text("Download CSV")').click()
  ])
  const name = await download.suggestedFilename()
  expect(name).toContain('esi_export_')
})



